"""
Argilla Dataset Management Utilities

This module provides utility functions for managing Argilla datasets, including:
- Creating, deleting, and updating datasets
- Managing dataset settings (fields, questions, metadata)
- Querying and listing datasets
- Publishing and managing dataset records
- Dataset import/export operations
- Exporting datasets with complete annotation data (NEW in 1.2.0)
- Batch exporting workspaces (NEW in 1.2.0)
- Conflict detection and review dataset generation (NEW in 1.3.0)

IMPORTANT: 所有数据集操作方法都支持 workspace 参数，避免误操作其他workspace的同名数据集
建议使用时明确指定 workspace，例如:
    manager.delete_dataset("dataset_name", workspace="my_workspace")

Author: Argilla Utils
Version: 1.3.0 (2025-12-04: 添加标注冲突检测功能，自动生成冲突审核数据集)
         1.2.0 (2025-11-28: 添加完整标注数据导出功能，支持CSV/Parquet，统计摘要)
"""

import os
import json
import shutil
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from dotenv import load_dotenv
import argilla as rg
from argilla._exceptions import ArgillaError
from argilla.records._dataset_records import RecordErrorHandling
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class ArgillaDatasetManager:
    """Manager class for Argilla dataset operations"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Argilla Dataset Manager
        
        Args:
            api_url: Argilla API URL (defaults to env variable or http://localhost:6900)
            api_key: Argilla API Key (defaults to env variable or owner.apikey)
        """
        load_dotenv()
        
        if api_url:
            os.environ["ARGILLA_API_URL"] = api_url
        elif "ARGILLA_API_URL" not in os.environ:
            os.environ["ARGILLA_API_URL"] = "http://localhost:6900"
            
        if api_key:
            os.environ["ARGILLA_API_KEY"] = api_key
        elif "ARGILLA_API_KEY" not in os.environ:
            os.environ["ARGILLA_API_KEY"] = "owner.apikey"
        
        self.client = rg.Argilla._get_default()
    
    def create_dataset(
        self,
        name: str,
        settings: rg.Settings,
        workspace: Optional[str] = None,
        overwrite: bool = False
    ) -> rg.Dataset:
        """
        Create a new dataset
        
        Args:
            name: Dataset name
            settings: Dataset settings (fields, questions, metadata)
            workspace: Workspace name (optional)
            overwrite: If True, delete existing dataset with same name
            
        Returns:
            Created Dataset object
        """
        # Check if dataset exists
        existing = self.get_dataset(name)
        if existing:
            if overwrite:
                print(f"⚠ Dataset '{name}' already exists. Deleting...")
                existing.delete()
            else:
                raise ArgillaError(f"Dataset '{name}' already exists. Use overwrite=True to replace.")
        
        # Create dataset
        dataset = rg.Dataset(name=name, settings=settings, workspace=workspace)
        dataset.create()
        
        print(f"✓ Dataset '{name}' created successfully")
        return dataset
    
    def delete_dataset(self, name: str, workspace: Optional[str] = None) -> bool:
        """
        Delete a dataset
        
        Args:
            name: Dataset name to delete
            workspace: Workspace name (optional, 用于区分不同workspace中的同名数据集)
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(name, workspace=workspace)
        if not dataset:
            workspace_info = f" in workspace '{workspace}'" if workspace else ""
            print(f"✗ Dataset '{name}'{workspace_info} not found")
            return False
        
        try:
            dataset.delete()
            workspace_info = f" from workspace '{workspace}'" if workspace else ""
            print(f"✓ Dataset '{name}'{workspace_info} deleted successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to delete dataset '{name}': {e}")
            return False
    
    def get_dataset(self, name: str, workspace: Optional[str] = None) -> Optional[rg.Dataset]:
        """
        Get dataset by name
        
        Args:
            name: Dataset name
            workspace: Workspace name (optional, 用于区分不同workspace中的同名数据集)
            
        Returns:
            Dataset object if found, None otherwise
        """
        try:
            dataset = self.client.datasets(name=name)
            # 如果指定了workspace，验证数据集是否属于该workspace
            if dataset and workspace:
                if hasattr(dataset, 'workspace') and dataset.workspace != workspace:
                    return None  # 数据集存在但不在指定的workspace中
            return dataset
        except Exception:
            return None
    
    def list_datasets(self, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all datasets
        
        Args:
            workspace: Filter by workspace name (optional)
            
        Returns:
            List of dictionaries containing dataset information
        """
        try:
            datasets = []
            for ds in self.client.datasets:
                # Filter by workspace if specified
                if workspace and hasattr(ds, 'workspace') and ds.workspace != workspace:
                    continue
                
                datasets.append({
                    "id": str(ds.id),
                    "name": ds.name,
                    "workspace": ds.workspace if hasattr(ds, 'workspace') else None,
                    "created_at": str(ds.created_at) if hasattr(ds, 'created_at') else None,
                    "updated_at": str(ds.updated_at) if hasattr(ds, 'updated_at') else None,
                })
            return datasets
        except Exception as e:
            print(f"✗ Failed to list datasets: {e}")
            return []
    
    def update_dataset_name(self, old_name: str, new_name: str, workspace: Optional[str] = None) -> bool:
        """
        Update dataset name
        
        Args:
            old_name: Current dataset name
            new_name: New dataset name
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(old_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{old_name}' not found")
            return False
        
        try:
            dataset.name = new_name
            dataset.update()
            print(f"✓ Dataset renamed from '{old_name}' to '{new_name}'")
            return True
        except Exception as e:
            print(f"✗ Failed to rename dataset: {e}")
            return False
    
    def add_records(
        self,
        dataset_name: str,
        records: List[Dict[str, Any]],
        mapping: Optional[Dict[str, str]] = None,
        on_error: str = "ignore",
        workspace: Optional[str] = None
    ) -> bool:
        """
        Add records to a dataset
        
        Args:
            dataset_name: Dataset name
            records: List of record dictionaries
            mapping: Mapping for suggestions (optional)
            on_error: Error handling strategy ('ignore', 'raise')
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        try:
            error_handling = RecordErrorHandling.IGNORE if on_error == "ignore" else RecordErrorHandling.RAISE
            dataset.records.log(records, mapping=mapping, on_error=error_handling)
            print(f"✓ Successfully added {len(records)} records to '{dataset_name}'")
            return True
        except Exception as e:
            print(f"✗ Failed to add records: {e}")
            return False
    
    def add_records_with_responses(
        self,
        dataset_name: str,
        records: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> bool:
        """
        添加记录并同时提交 Response（状态为 submitted）
        
        Args:
            dataset_name: 数据集名称
            records: 记录列表，每条记录包含:
                - fields: dict, 必需，字段内容 (例如: {"text": "内容"})
                - responses: dict, 必需，Response内容 (例如: {"label": "positive"})
                - metadata: dict, 可选，元数据
                - suggestions: dict, 可选，建议标注
            user_id: 用户ID (可选，默认为当前用户)
            workspace: 工作区名称 (可选)
            
        Returns:
            True if successful, False otherwise
            
        Example:
            records = [
                {
                    "fields": {"text": "这是一条测试文本"},
                    "responses": {"label": "positive"},  # 会自动设置为 submitted 状态
                    "metadata": {"source": "manual"},
                    "suggestions": {"label": "positive"}  # 可选
                },
                {
                    "fields": {"text": "另一条测试文本"},
                    "responses": {"label": "negative"}
                }
            ]
            manager.add_records_with_responses("my_dataset", records)
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        # 获取当前用户ID（如果未指定）
        if user_id is None:
            user_id = self.client.me.id
        
        try:
            # 构建 Argilla Record 对象列表
            argilla_records = []
            
            for record_data in records:
                fields = record_data.get("fields", {})
                responses_data = record_data.get("responses", {})
                metadata = record_data.get("metadata")
                suggestions_data = record_data.get("suggestions")
                
                # 验证必需字段
                if not fields:
                    print(f"⚠ Warning: Record missing 'fields', skipping")
                    continue
                
                # 构建 responses (status=submitted)
                responses = []
                if responses_data:
                    for question_name, answer_value in responses_data.items():
                        responses.append(
                            rg.Response(
                                question_name=question_name,
                                value=answer_value,
                                user_id=user_id,
                                status="submitted"  # 关键：设置为 submitted
                            )
                        )
                
                # 构建 suggestions
                suggestions = []
                if suggestions_data:
                    for question_name, suggestion_value in suggestions_data.items():
                        suggestions.append(
                            rg.Suggestion(
                                question_name=question_name,
                                value=suggestion_value
                            )
                        )
                
                # 创建 Record 对象
                argilla_record = rg.Record(
                    fields=fields,
                    metadata=metadata,
                    responses=responses if responses else None,
                    suggestions=suggestions if suggestions else None
                )
                
                argilla_records.append(argilla_record)
            
            # 批量添加记录
            if argilla_records:
                dataset.records.log(argilla_records)
                print(f"✓ Successfully added {len(argilla_records)} records with submitted responses to '{dataset_name}'")
                return True
            else:
                print(f"⚠ No valid records to add")
                return False
                
        except Exception as e:
            print(f"✗ Failed to add records with responses: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_records(self, dataset_name: str, record_ids: List[str], workspace: Optional[str] = None) -> bool:
        """
        Delete specific records from a dataset
        
        Args:
            dataset_name: Dataset name
            record_ids: List of record IDs to delete
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        try:
            for record_id in record_ids:
                # Find and delete record
                for record in dataset.records:
                    if str(record.id) == str(record_id):
                        record.delete()
                        break
            
            print(f"✓ Successfully deleted {len(record_ids)} records from '{dataset_name}'")
            return True
        except Exception as e:
            print(f"✗ Failed to delete records: {e}")
            return False
    
    def clear_all_records(self, dataset_name: str, workspace: Optional[str] = None) -> bool:
        """
        Clear all records from a dataset (keeps dataset structure)
        
        Args:
            dataset_name: Dataset name
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        try:
            # Delete all records
            count = 0
            for record in dataset.records:
                record.delete()
                count += 1
            
            print(f"✓ Cleared {count} records from '{dataset_name}'")
            return True
        except Exception as e:
            print(f"✗ Failed to clear records: {e}")
            return False
    
    def get_record_count(self, dataset_name: str, workspace: Optional[str] = None) -> int:
        """
        Get total number of records in a dataset
        
        Args:
            dataset_name: Dataset name
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            Number of records, or -1 if error
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return -1
        
        try:
            count = sum(1 for _ in dataset.records)
            return count
        except Exception as e:
            print(f"✗ Failed to count records: {e}")
            return -1
    
    def export_dataset(
        self,
        dataset_name: str,
        export_path: str,
        format: str = "disk",
        ensure_chinese: bool = True,
        workspace: Optional[str] = None
    ) -> bool:
        """
        Export dataset to local storage
        
        Args:
            dataset_name: Dataset name
            export_path: Local path to export to
            format: Export format ('disk' for Argilla native format)
            ensure_chinese: If True, fix JSON encoding to display Chinese properly
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        try:
            # Clean old export if exists
            if os.path.exists(export_path):
                shutil.rmtree(export_path)
            
            # Export to disk
            dataset.to_disk(export_path)
            
            # Fix Chinese encoding if requested
            if ensure_chinese:
                records_path = os.path.join(export_path, "records.json")
                if os.path.exists(records_path):
                    with open(records_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    with open(records_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"✓ Dataset '{dataset_name}' exported to '{export_path}'")
            return True
        except Exception as e:
            print(f"✗ Failed to export dataset: {e}")
            return False
    
    def import_dataset(
        self,
        import_path: str,
        new_name: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> Optional[rg.Dataset]:
        """
        Import dataset from local storage
        
        Args:
            import_path: Local path to import from
            new_name: New dataset name (optional, uses original name if not provided)
            workspace: Target workspace (optional)
            
        Returns:
            Imported Dataset object, or None if failed
        """
        try:
            # Load from disk
            dataset = rg.Dataset.from_disk(import_path)
            
            # Rename if requested
            if new_name:
                dataset.name = new_name
            
            # Set workspace if requested
            if workspace:
                dataset.workspace = workspace
            
            # Create on server
            dataset.create()
            
            print(f"✓ Dataset imported successfully as '{dataset.name}'")
            return dataset
        except Exception as e:
            print(f"✗ Failed to import dataset: {e}")
            return None
    
    def clone_dataset(
        self,
        source_name: str,
        target_name: str,
        include_records: bool = True,
        source_workspace: Optional[str] = None,
        target_workspace: Optional[str] = None
    ) -> Optional[rg.Dataset]:
        """
        Clone a dataset (with or without records)
        
        Args:
            source_name: Source dataset name
            target_name: Target dataset name
            include_records: If True, copy records as well
            source_workspace: Source workspace (optional, 避免误操作其他workspace的同名数据集)
            target_workspace: Target workspace (optional, 目标workspace，可以跨workspace克隆)
            
        Returns:
            Cloned Dataset object, or None if failed
        """
        source = self.get_dataset(source_name, workspace=source_workspace)
        if not source:
            print(f"✗ Source dataset '{source_name}' not found")
            return None
        
        try:
            # Create new dataset with same settings
            target = rg.Dataset(
                name=target_name,
                settings=source.settings,
                workspace=target_workspace
            )
            target.create()
            
            # Copy records if requested
            if include_records:
                records = []
                for record in source.records:
                    # Convert record to dict
                    rec_dict = {
                        "fields": record.fields,
                        "metadata": record.metadata,
                    }
                    records.append(rec_dict)
                
                if records:
                    target.records.log(records)
            
            print(f"✓ Dataset cloned from '{source_name}' to '{target_name}'")
            return target
        except Exception as e:
            print(f"✗ Failed to clone dataset: {e}")
            return None
    
    def get_dataset_info(self, dataset_name: str, workspace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a dataset
        
        Args:
            dataset_name: Dataset name
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            Dictionary with dataset information
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return None
        
        try:
            # Count records
            record_count = sum(1 for _ in dataset.records)
            
            # Get settings info
            settings = dataset.settings
            field_names = [f.name for f in settings.fields] if settings.fields else []
            question_names = [q.name for q in settings.questions] if settings.questions else []
            metadata_names = [m.name for m in settings.metadata] if settings.metadata else []
            
            info = {
                "id": str(dataset.id),
                "name": dataset.name,
                "workspace": dataset.workspace if hasattr(dataset, 'workspace') else None,
                "record_count": record_count,
                "fields": field_names,
                "questions": question_names,
                "metadata": metadata_names,
                "created_at": str(dataset.created_at) if hasattr(dataset, 'created_at') else None,
                "updated_at": str(dataset.updated_at) if hasattr(dataset, 'updated_at') else None,
            }
            
            return info
        except Exception as e:
            print(f"✗ Failed to get dataset info: {e}")
            return None
    
    def _export_dataset_settings(self, dataset: rg.Dataset) -> Dict[str, Any]:
        """
        导出数据集的 settings 信息（字段定义、问题定义、metadata定义）
        
        Args:
            dataset: Dataset 对象
            
        Returns:
            包含 settings 信息的字典
        """
        settings_info = {
            "dataset_name": dataset.name,
            "fields": [],
            "questions": [],
            "metadata_properties": [],
            "vectors": []
        }
        
        # 导出字段定义
        if dataset.settings and dataset.settings.fields:
            for field in dataset.settings.fields:
                field_info = {
                    "name": field.name,
                    "title": field.title if hasattr(field, 'title') else field.name,
                    "type": field.__class__.__name__,  # TextField, etc.
                    "required": field.required if hasattr(field, 'required') else False,
                }
                
                # 添加特定字段类型的属性
                if hasattr(field, 'use_markdown'):
                    field_info["use_markdown"] = field.use_markdown
                    
                settings_info["fields"].append(field_info)
        
        # 导出问题定义
        if dataset.settings and dataset.settings.questions:
            for question in dataset.settings.questions:
                question_info = {
                    "name": question.name,
                    "title": question.title if hasattr(question, 'title') else question.name,
                    "type": question.__class__.__name__,  # LabelQuestion, TextQuestion, etc.
                    "required": question.required if hasattr(question, 'required') else False,
                }
                
                # 添加特定问题类型的属性
                if hasattr(question, 'labels'):  # LabelQuestion, MultiLabelQuestion
                    question_info["labels"] = question.labels
                if hasattr(question, 'visible_labels'):
                    question_info["visible_labels"] = question.visible_labels
                if hasattr(question, 'use_markdown'):
                    question_info["use_markdown"] = question.use_markdown
                if hasattr(question, 'description'):
                    question_info["description"] = question.description
                    
                settings_info["questions"].append(question_info)
        
        # 导出 metadata 属性定义
        if dataset.settings and dataset.settings.metadata:
            for metadata_prop in dataset.settings.metadata:
                meta_info = {
                    "name": metadata_prop.name,
                    "title": metadata_prop.title if hasattr(metadata_prop, 'title') else metadata_prop.name,
                    "type": metadata_prop.__class__.__name__,  # IntegerMetadataProperty, FloatMetadataProperty, etc.
                }
                
                # 添加特定 metadata 类型的属性
                if hasattr(metadata_prop, 'min'):
                    meta_info["min"] = metadata_prop.min
                if hasattr(metadata_prop, 'max'):
                    meta_info["max"] = metadata_prop.max
                if hasattr(metadata_prop, 'values'):  # TermsMetadataProperty
                    meta_info["values"] = metadata_prop.values
                    
                settings_info["metadata_properties"].append(meta_info)
        
        # 导出 vectors 定义（如果有）
        if dataset.settings and dataset.settings.vectors:
            for vector in dataset.settings.vectors:
                vector_info = {
                    "name": vector.name,
                    "title": vector.title if hasattr(vector, 'title') else vector.name,
                    "dimensions": vector.dimensions if hasattr(vector, 'dimensions') else None
                }
                settings_info["vectors"].append(vector_info)
        
        return settings_info
    
    def export_dataset_with_annotations(
        self,
        dataset_name: str,
        export_dir: str,
        format: str = "csv",
        workspace: Optional[str] = None,
        include_summary: bool = True
    ) -> bool:
        """
        导出数据集及完整的标注数据（包括标注老师的标注结果）
        
        Args:
            dataset_name: 数据集名称
            export_dir: 导出目录
            format: 导出格式 ('csv' 或 'parquet')
            workspace: 工作区名称 (可选)
            include_summary: 是否包含统计摘要
            
        Returns:
            True if successful, False otherwise
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        if format not in ["csv", "parquet"]:
            print(f"✗ Invalid format '{format}'. Use 'csv' or 'parquet'")
            return False
        
        if format == "parquet" and not PARQUET_AVAILABLE:
            print("✗ Parquet format requires pyarrow. Install with: pip install pyarrow")
            return False
        
        try:
            # 创建导出目录
            os.makedirs(export_dir, exist_ok=True)
            
            # 收集所有数据
            all_records = []
            annotation_stats = defaultdict(lambda: {"count": 0, "responses": Counter()})
            
            for record in dataset.records:
                # 基础记录信息
                base_record = {
                    "record_id": str(record.id),
                    "created_at": str(record.inserted_at) if hasattr(record, 'inserted_at') else None,
                    "updated_at": str(record.updated_at) if hasattr(record, 'updated_at') else None,
                }
                
                # 添加字段内容
                for field_name, field_value in record.fields.items():
                    base_record[f"field_{field_name}"] = field_value
                
                # 添加元数据
                if record.metadata:
                    for meta_key, meta_value in record.metadata.items():
                        base_record[f"metadata_{meta_key}"] = meta_value
                
                # 处理标注结果 (responses)
                if record.responses:
                    for resp in record.responses:
                        # 创建一条记录（每个 response 一条）
                        response_record = base_record.copy()
                        
                        # 标注者信息
                        annotator_id = str(resp.user_id) if hasattr(resp, 'user_id') else "unknown"
                        response_record["annotator_id"] = annotator_id
                        
                        # 获取标注者用户名
                        try:
                            user = self.client.users(annotator_id)
                            response_record["annotator_username"] = user.username if user else annotator_id
                        except:
                            response_record["annotator_username"] = annotator_id
                        
                        # 标注问题和答案
                        response_record["question_name"] = resp.question_name if hasattr(resp, 'question_name') else None
                        response_record["response_value"] = resp.value if hasattr(resp, 'value') else None
                        response_record["response_status"] = resp.status if hasattr(resp, 'status') else None
                        response_record["response_created_at"] = str(resp.inserted_at) if hasattr(resp, 'inserted_at') else None
                        response_record["response_updated_at"] = str(resp.updated_at) if hasattr(resp, 'updated_at') else None
                        
                        all_records.append(response_record)
                        
                        # 统计信息
                        username = response_record["annotator_username"]
                        annotation_stats[username]["count"] += 1
                        annotation_stats[username]["responses"][response_record["response_value"]] += 1
                
                else:
                    # 没有标注的记录也要导出
                    base_record["annotator_id"] = None
                    base_record["annotator_username"] = None
                    base_record["question_name"] = None
                    base_record["response_value"] = None
                    base_record["response_status"] = None
                    base_record["response_created_at"] = None
                    base_record["response_updated_at"] = None
                    all_records.append(base_record)
            
            if not all_records:
                print(f"⚠ No records found in dataset '{dataset_name}'")
                return False
            
            # 创建 DataFrame
            df = pd.DataFrame(all_records)
            
            # 导出数据
            timestamp = datetime.now().strftime("%Y%m%d")
            dataset_safe_name = dataset_name.replace("/", "_").replace("\\", "_")
            
            if format == "csv":
                csv_path = os.path.join(export_dir, f"{dataset_safe_name}_{timestamp}.csv")
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compatibility
                print(f"✓ Exported {len(all_records)} records to CSV: {csv_path}")
            
            elif format == "parquet":
                parquet_path = os.path.join(export_dir, f"{dataset_safe_name}_{timestamp}.parquet")
                df.to_parquet(parquet_path, index=False, engine="pyarrow")
                print(f"✓ Exported {len(all_records)} records to Parquet: {parquet_path}")
            
            # 导出 dataset settings (字段定义、问题定义、metadata 定义)
            settings_info = self._export_dataset_settings(dataset)
            settings_path = os.path.join(export_dir, f"{dataset_safe_name}_{timestamp}_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings_info, f, indent=4, ensure_ascii=False)
            print(f"✓ Exported dataset settings to: {settings_path}")
            
            # 导出统计摘要
            if include_summary:
                summary = {
                    "dataset_name": dataset_name,
                    "workspace": workspace,
                    "export_date": datetime.now().isoformat(),
                    "total_records": len(all_records),
                    "unique_records": len(df["record_id"].unique()),
                    "annotators": [],
                    "settings_file": f"{dataset_safe_name}_{timestamp}_settings.json"
                }
                
                for username, stats in annotation_stats.items():
                    annotator_info = {
                        "username": username,
                        "total_annotations": stats["count"],
                        "response_distribution": dict(stats["responses"])
                    }
                    summary["annotators"].append(annotator_info)
                
                summary_path = os.path.join(export_dir, f"{dataset_safe_name}_{timestamp}_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4, ensure_ascii=False)
                print(f"✓ Exported summary to: {summary_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to export dataset with annotations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def export_workspace_datasets(
        self,
        workspace: str,
        export_base_dir: str,
        format: str = "csv",
        include_summary: bool = True
    ) -> bool:
        """
        导出整个 workspace 中所有数据集的标注数据（按天组织）
        
        Args:
            workspace: 工作区名称
            export_base_dir: 导出根目录
            format: 导出格式 ('csv' 或 'parquet')
            include_summary: 是否包含统计摘要
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 创建按日期命名的目录
            today = datetime.now().strftime("%Y-%m-%d")
            export_dir = os.path.join(export_base_dir, workspace, today)
            os.makedirs(export_dir, exist_ok=True)
            
            # 获取 workspace 中的所有数据集
            datasets = self.list_datasets(workspace=workspace)
            
            if not datasets:
                print(f"✗ No datasets found in workspace '{workspace}'")
                return False
            
            print(f"\n{'='*60}")
            print(f"Exporting workspace '{workspace}' - {len(datasets)} datasets")
            print(f"Export directory: {export_dir}")
            print(f"{'='*60}\n")
            
            # 导出每个数据集
            success_count = 0
            failed_datasets = []
            
            for ds_info in datasets:
                dataset_name = ds_info["name"]
                print(f"\n[{success_count + 1}/{len(datasets)}] Processing: {dataset_name}")
                
                success = self.export_dataset_with_annotations(
                    dataset_name=dataset_name,
                    export_dir=export_dir,
                    format=format,
                    workspace=workspace,
                    include_summary=include_summary
                )
                
                if success:
                    success_count += 1
                else:
                    failed_datasets.append(dataset_name)
            
            # 生成 workspace 级别的摘要
            workspace_summary = {
                "workspace": workspace,
                "export_date": datetime.now().isoformat(),
                "total_datasets": len(datasets),
                "successfully_exported": success_count,
                "failed_exports": failed_datasets,
                "format": format,
                "datasets": [ds["name"] for ds in datasets]
            }
            
            summary_path = os.path.join(export_dir, f"workspace_summary_{today}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(workspace_summary, f, indent=4, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"✓ Workspace export completed!")
            print(f"  - Successfully exported: {success_count}/{len(datasets)} datasets")
            if failed_datasets:
                print(f"  - Failed: {', '.join(failed_datasets)}")
            print(f"  - Summary saved to: {summary_path}")
            print(f"{'='*60}\n")
            
            return success_count > 0
            
        except Exception as e:
            print(f"✗ Failed to export workspace datasets: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_dataset_from_settings(
        self,
        settings_file: str,
        dataset_name: str,
        workspace: Optional[str] = None
    ) -> Optional[rg.Dataset]:
        """
        根据 settings JSON 文件创建数据集
        
        Args:
            settings_file: settings JSON 文件路径
            dataset_name: 数据集名称
            workspace: 工作区名称（可选）
            
        Returns:
            创建的 Dataset 对象，失败返回 None
        """
        try:
            # 读取 settings 文件
            with open(settings_file, "r", encoding="utf-8") as f:
                settings_info = json.load(f)
            
            # 创建 Settings 对象
            settings = rg.Settings()
            
            # 添加字段定义
            for field_info in settings_info.get("fields", []):
                field_type = field_info.get("type", "TextField")
                
                if field_type == "TextField":
                    field = rg.TextField(
                        name=field_info["name"],
                        title=field_info.get("title", field_info["name"]),
                        required=field_info.get("required", False),
                        use_markdown=field_info.get("use_markdown", False)
                    )
                elif field_type == "ChatField":
                    field = rg.ChatField(
                        name=field_info["name"],
                        title=field_info.get("title", field_info["name"]),
                        required=field_info.get("required", False),
                        use_markdown=field_info.get("use_markdown", False)
                    )
                elif field_type == "ImageField":
                    field = rg.ImageField(
                        name=field_info["name"],
                        title=field_info.get("title", field_info["name"]),
                        required=field_info.get("required", False)
                    )
                else:
                    # 默认使用 TextField
                    field = rg.TextField(
                        name=field_info["name"],
                        title=field_info.get("title", field_info["name"]),
                        required=field_info.get("required", False)
                    )
                
                settings.fields.append(field)
            
            # 添加问题定义
            for question_info in settings_info.get("questions", []):
                question_type = question_info.get("type", "LabelQuestion")
                
                if question_type == "LabelQuestion":
                    question = rg.LabelQuestion(
                        name=question_info["name"],
                        title=question_info.get("title", question_info["name"]),
                        labels=question_info.get("labels", []),
                        required=question_info.get("required", False),
                        description=question_info.get("description")
                    )
                elif question_type == "MultiLabelQuestion":
                    question = rg.MultiLabelQuestion(
                        name=question_info["name"],
                        title=question_info.get("title", question_info["name"]),
                        labels=question_info.get("labels", []),
                        required=question_info.get("required", False),
                        description=question_info.get("description")
                    )
                elif question_type == "TextQuestion":
                    question = rg.TextQuestion(
                        name=question_info["name"],
                        title=question_info.get("title", question_info["name"]),
                        required=question_info.get("required", False),
                        description=question_info.get("description"),
                        use_markdown=question_info.get("use_markdown", False)
                    )
                elif question_type == "RatingQuestion":
                    question = rg.RatingQuestion(
                        name=question_info["name"],
                        title=question_info.get("title", question_info["name"]),
                        values=question_info.get("values", [1, 2, 3, 4, 5]),
                        required=question_info.get("required", False),
                        description=question_info.get("description")
                    )
                elif question_type == "RankingQuestion":
                    question = rg.RankingQuestion(
                        name=question_info["name"],
                        title=question_info.get("title", question_info["name"]),
                        values=question_info.get("values", []),
                        required=question_info.get("required", False),
                        description=question_info.get("description")
                    )
                elif question_type == "SpanQuestion":
                    question = rg.SpanQuestion(
                        name=question_info["name"],
                        title=question_info.get("title", question_info["name"]),
                        field=question_info.get("field"),
                        labels=question_info.get("labels", []),
                        required=question_info.get("required", False),
                        description=question_info.get("description")
                    )
                else:
                    # 默认使用 TextQuestion
                    question = rg.TextQuestion(
                        name=question_info["name"],
                        title=question_info.get("title", question_info["name"]),
                        required=question_info.get("required", False)
                    )
                
                settings.questions.append(question)
            
            # 添加 metadata 属性定义
            for meta_info in settings_info.get("metadata_properties", []):
                meta_type = meta_info.get("type", "TermsMetadataProperty")
                
                if meta_type == "IntegerMetadataProperty":
                    metadata_prop = rg.IntegerMetadataProperty(
                        name=meta_info["name"],
                        title=meta_info.get("title", meta_info["name"]),
                        min=meta_info.get("min"),
                        max=meta_info.get("max")
                    )
                elif meta_type == "FloatMetadataProperty":
                    metadata_prop = rg.FloatMetadataProperty(
                        name=meta_info["name"],
                        title=meta_info.get("title", meta_info["name"]),
                        min=meta_info.get("min"),
                        max=meta_info.get("max")
                    )
                elif meta_type == "TermsMetadataProperty":
                    metadata_prop = rg.TermsMetadataProperty(
                        name=meta_info["name"],
                        title=meta_info.get("title", meta_info["name"]),
                        values=meta_info.get("values")
                    )
                else:
                    # 默认使用 TermsMetadataProperty
                    metadata_prop = rg.TermsMetadataProperty(
                        name=meta_info["name"],
                        title=meta_info.get("title", meta_info["name"])
                    )
                
                settings.metadata.append(metadata_prop)
            
            # 添加 vectors 定义（如果有）
            for vector_info in settings_info.get("vectors", []):
                vector = rg.VectorField(
                    name=vector_info["name"],
                    title=vector_info.get("title", vector_info["name"]),
                    dimensions=vector_info.get("dimensions")
                )
                settings.vectors.append(vector)
            
            # 创建数据集
            dataset = rg.Dataset(
                name=dataset_name,
                settings=settings,
                workspace=workspace
            )
            dataset.create()
            
            print(f"✓ Created dataset '{dataset_name}' with imported settings")
            return dataset
            
        except Exception as e:
            print(f"✗ Failed to create dataset from settings: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def import_dataset_with_annotations(
        self,
        import_file: str,
        dataset_name: Optional[str] = None,
        workspace: Optional[str] = None,
        create_if_not_exists: bool = True,
        settings_file: Optional[str] = None
    ) -> bool:
        """
        从 CSV 或 Parquet 文件导入数据集及标注数据
        
        Args:
            import_file: 导入文件路径 (CSV 或 Parquet)
            dataset_name: 目标数据集名称（可选，默认从文件名提取）
            workspace: 目标工作区名称（可选）
            create_if_not_exists: 如果数据集不存在是否创建
            settings_file: settings JSON 文件路径（可选，用于自动创建数据集）
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 读取文件
            if import_file.endswith(".csv"):
                df = pd.read_csv(import_file, encoding="utf-8")
            elif import_file.endswith(".parquet"):
                df = pd.read_parquet(import_file)
            else:
                print(f"✗ Unsupported file format. Use .csv or .parquet")
                return False
            
            # 提取数据集名称
            if not dataset_name:
                filename = os.path.basename(import_file)
                # 移除时间戳和扩展名
                parts = filename.rsplit("_", 1)
                if len(parts) > 1 and parts[1].replace(".csv", "").replace(".parquet", "").isdigit():
                    dataset_name = parts[0]
                else:
                    dataset_name = filename.split(".")[0]
            
            print(f"Importing {len(df)} records to dataset '{dataset_name}'...")
            
            # 检查数据集是否存在
            dataset = self.get_dataset(dataset_name, workspace=workspace)
            
            if not dataset and not create_if_not_exists:
                print(f"✗ Dataset '{dataset_name}' not found and create_if_not_exists=False")
                return False
            
            # 如果数据集不存在，尝试从 settings 文件创建
            if not dataset and create_if_not_exists:
                # 如果没有指定 settings_file，尝试自动查找
                if not settings_file:
                    # 查找同名的 settings 文件
                    import_dir = os.path.dirname(import_file)
                    base_name = os.path.basename(import_file).rsplit("_", 1)[0]
                    
                    # 尝试多种可能的 settings 文件名
                    possible_settings = [
                        os.path.join(import_dir, f"{base_name}_settings.json"),
                        os.path.join(import_dir, f"{dataset_name}_settings.json"),
                    ]
                    
                    # 也尝试从时间戳文件名中提取
                    for f in os.listdir(import_dir):
                        if f.endswith("_settings.json") and base_name in f:
                            possible_settings.append(os.path.join(import_dir, f))
                    
                    for settings_path in possible_settings:
                        if os.path.exists(settings_path):
                            settings_file = settings_path
                            print(f"ℹ Found settings file: {settings_file}")
                            break
                
                # 如果找到了 settings 文件，使用它创建数据集
                if settings_file and os.path.exists(settings_file):
                    dataset = self._create_dataset_from_settings(
                        settings_file=settings_file,
                        dataset_name=dataset_name,
                        workspace=workspace
                    )
                    if not dataset:
                        print(f"✗ Failed to create dataset from settings file")
                        return False
                else:
                    print(f"✗ Dataset '{dataset_name}' not found and no settings file provided")
                    print(f"  Please provide a settings file or create the dataset manually")
                    return False
            
            # 按 record_id 分组（因为每个 record 可能有多个 response）
            grouped = df.groupby("record_id")
            
            records_to_add = []
            
            for record_id, group in grouped:
                # 提取字段
                fields = {}
                metadata = {}
                responses_data = {}
                
                first_row = group.iloc[0]
                
                # 提取 field_ 开头的列
                for col in df.columns:
                    if col.startswith("field_"):
                        field_name = col.replace("field_", "")
                        fields[field_name] = first_row[col]
                    elif col.startswith("metadata_"):
                        meta_name = col.replace("metadata_", "")
                        metadata[meta_name] = first_row[col]
                
                # 提取所有标注结果（可能有多个标注者）
                # 注意：这里我们只取第一个标注结果用于导入
                # 如果有多个标注者，可能需要特殊处理
                if pd.notna(first_row.get("response_value")):
                    question_name = first_row.get("question_name")
                    if question_name:
                        responses_data[question_name] = first_row["response_value"]
                
                # 构建记录
                record_data = {
                    "fields": fields,
                    "metadata": metadata if metadata else None,
                    "responses": responses_data if responses_data else None
                }
                
                records_to_add.append(record_data)
            
            # 批量添加记录
            if records_to_add:
                # 使用 add_records_with_responses 方法
                success = self.add_records_with_responses(
                    dataset_name=dataset_name,
                    records=records_to_add,
                    workspace=workspace
                )
                
                if success:
                    print(f"✓ Successfully imported {len(records_to_add)} records to '{dataset_name}'")
                    return True
                else:
                    print(f"✗ Failed to import records")
                    return False
            else:
                print(f"⚠ No valid records found in import file")
                return False
                
        except Exception as e:
            print(f"✗ Failed to import dataset with annotations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect_conflicts_for_dataset(
        self,
        dataset_name: str,
        label_question: str = "label",
        cot_question: str = "COT",
        workspace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        检测数据集中的标注冲突（两位标注者的label不一致）
        
        Args:
            dataset_name: 数据集名称
            label_question: 标签问题名称 (默认 "label")
            cot_question: COT问题名称 (默认 "COT")
            workspace: 工作区名称 (可选)
            
        Returns:
            冲突记录列表，每条包含:
            {
                "record_id": 原始记录ID,
                "original_dataset": 原始数据集名称,
                "fields": 原始字段内容,
                "metadata": 原始元数据,
                "annotator_1": {username, label, cot, time},
                "annotator_2": {username, label, cot, time},
                "conflict_detected_at": 检测时间
            }
        """
        dataset = self.get_dataset(dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return []
        
        # 获取用户信息映射
        user_mapping = {}
        try:
            for user in self.client.users:
                user_mapping[str(user.id)] = user.username
        except:
            pass
        
        conflicts = []
        
        try:
            for record in dataset.records:
                if not record.responses:
                    continue
                
                # 收集每个用户对label问题的响应
                user_responses = {}  # {user_id: {label, cot, time}}
                
                for resp in record.responses:
                    if resp.status != "submitted":
                        continue
                    
                    user_id = str(resp.user_id)
                    
                    # 初始化用户响应记录
                    if user_id not in user_responses:
                        user_responses[user_id] = {
                            "username": user_mapping.get(user_id, user_id),
                            "label": None,
                            "cot": None,
                            "time": None
                        }
                    
                    # 收集label和COT
                    if resp.question_name == label_question:
                        user_responses[user_id]["label"] = resp.value
                        # 获取响应时间
                        if hasattr(resp, 'updated_at') and resp.updated_at:
                            user_responses[user_id]["time"] = str(resp.updated_at)
                        elif hasattr(resp, 'inserted_at') and resp.inserted_at:
                            user_responses[user_id]["time"] = str(resp.inserted_at)
                    
                    elif resp.question_name == cot_question:
                        user_responses[user_id]["cot"] = resp.value
                
                # 过滤出有label标注的用户
                labeled_users = {uid: data for uid, data in user_responses.items() 
                                if data["label"] is not None}
                
                # 至少需要两个人标注了label才能检测冲突
                if len(labeled_users) < 2:
                    continue
                
                # 获取前两个标注者（按时间排序或任意取两个）
                user_list = list(labeled_users.items())[:2]
                user1_id, user1_data = user_list[0]
                user2_id, user2_data = user_list[1]
                
                # 检查label是否冲突
                if user1_data["label"] != user2_data["label"]:
                    conflict_info = {
                        "record_id": str(record.id),
                        "original_dataset": dataset_name,
                        "fields": dict(record.fields) if record.fields else {},
                        "metadata": dict(record.metadata) if record.metadata else {},
                        "annotator_1": user1_data,
                        "annotator_2": user2_data,
                        "conflict_detected_at": datetime.now().isoformat()
                    }
                    conflicts.append(conflict_info)
            
            print(f"✓ Found {len(conflicts)} conflicts in dataset '{dataset_name}'")
            return conflicts
            
        except Exception as e:
            print(f"✗ Failed to detect conflicts in dataset '{dataset_name}': {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_conflict_review_dataset(
        self,
        conflict_dataset_name: str,
        labels: List[str],
        workspace: Optional[str] = None
    ) -> Optional[rg.Dataset]:
        """
        创建冲突审核数据集
        
        Args:
            conflict_dataset_name: 冲突数据集名称
            labels: 可选的标签列表
            workspace: 工作区名称
            
        Returns:
            创建的数据集对象
        """
        try:
            # 定义字段
            fields = [
                rg.TextField(name="original_text", title="原始文本", use_markdown=True),
                rg.TextField(name="annotation_comparison", title="标注对比", use_markdown=True),
                rg.TextField(name="conflict_info", title="冲突信息", use_markdown=True),
            ]
            
            # 定义问题 - 审核者选择正确的标签
            questions = [
                rg.LabelQuestion(
                    name="correct_label",
                    title="正确的分类标签",
                    labels=labels,
                    required=True,
                    description="请根据原始文本和两位标注者的判断，选择正确的分类标签"
                ),
                rg.TextQuestion(
                    name="review_reason",
                    title="审核理由",
                    required=False,
                    description="请简要说明选择该标签的理由（可选）"
                ),
            ]
            
            # 定义元数据
            metadata_props = [
                rg.TermsMetadataProperty(name="original_dataset", title="原始数据集"),
                rg.TermsMetadataProperty(name="original_record_id", title="原始记录ID"),
                rg.TermsMetadataProperty(name="annotator_1_username", title="标注者1"),
                rg.TermsMetadataProperty(name="annotator_2_username", title="标注者2"),
                rg.TermsMetadataProperty(name="label_1", title="标注者1的标签"),
                rg.TermsMetadataProperty(name="label_2", title="标注者2的标签"),
            ]
            
            # 创建设置
            settings = rg.Settings(
                fields=fields,
                questions=questions,
                metadata=metadata_props
            )
            
            # 创建数据集
            dataset = rg.Dataset(
                name=conflict_dataset_name,
                settings=settings,
                workspace=workspace
            )
            dataset.create()
            
            print(f"✓ Created conflict review dataset: '{conflict_dataset_name}'")
            return dataset
            
        except Exception as e:
            print(f"✗ Failed to create conflict review dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_existing_conflict_record_ids(
        self,
        conflict_dataset: rg.Dataset
    ) -> set:
        """
        获取冲突数据集中已存在的原始记录ID集合
        
        Args:
            conflict_dataset: 冲突数据集
            
        Returns:
            已存在的原始记录ID集合
        """
        existing_ids = set()
        try:
            for record in conflict_dataset.records:
                if record.metadata and "original_record_id" in record.metadata:
                    existing_ids.add(record.metadata["original_record_id"])
        except Exception as e:
            print(f"⚠ Warning: Failed to get existing record IDs: {e}")
        
        return existing_ids
    
    def _format_annotation_comparison_table(
        self, 
        annotator_1: Dict[str, Any], 
        annotator_2: Dict[str, Any]
    ) -> str:
        """
        格式化两位标注者的信息为HTML表格
        
        Args:
            annotator_1: 标注者1数据 {username, label, cot, time}
            annotator_2: 标注者2数据 {username, label, cot, time}
            
        Returns:
            格式化的HTML表格
        """
        username_1 = annotator_1.get("username", "未知")
        username_2 = annotator_2.get("username", "未知")
        label_1 = annotator_1.get("label", "无")
        label_2 = annotator_2.get("label", "无")
        cot_1 = annotator_1.get("cot") or "未填写"
        cot_2 = annotator_2.get("cot") or "未填写"
        time_1 = annotator_1.get("time", "未知")
        time_2 = annotator_2.get("time", "未知")
        
        table = f"""<table style="width:100%; border-collapse:collapse; margin:10px 0;">
  <thead>
    <tr style="background-color:#f5f5f5;">
      <th style="border:1px solid #ddd; padding:10px; text-align:left; width:80px;"></th>
      <th style="border:1px solid #ddd; padding:10px; text-align:center;"><strong>{username_1}</strong></th>
      <th style="border:1px solid #ddd; padding:10px; text-align:center;"><strong>{username_2}</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #ddd; padding:10px;"><strong>Label</strong></td>
      <td style="border:1px solid #ddd; padding:10px; text-align:center;"><code style="background:#e7f3ff; padding:2px 6px; border-radius:3px;">{label_1}</code></td>
      <td style="border:1px solid #ddd; padding:10px; text-align:center;"><code style="background:#fff3e7; padding:2px 6px; border-radius:3px;">{label_2}</code></td>
    </tr>
    <tr style="background-color:#fafafa;">
      <td style="border:1px solid #ddd; padding:10px;"><strong>COT</strong></td>
      <td style="border:1px solid #ddd; padding:10px;">{cot_1}</td>
      <td style="border:1px solid #ddd; padding:10px;">{cot_2}</td>
    </tr>
    <tr>
      <td style="border:1px solid #ddd; padding:10px;"><strong>时间</strong></td>
      <td style="border:1px solid #ddd; padding:10px; font-size:12px; color:#666;">{time_1}</td>
      <td style="border:1px solid #ddd; padding:10px; font-size:12px; color:#666;">{time_2}</td>
    </tr>
  </tbody>
</table>"""
        return table
    
    def create_or_update_conflict_dataset(
        self,
        workspace: str,
        conflict_dataset_name: Optional[str] = None,
        label_question: str = "label",
        cot_question: str = "COT",
        target_datasets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        检测workspace下所有数据集的标注冲突，并创建/更新冲突审核数据集
        
        Args:
            workspace: 工作区名称
            conflict_dataset_name: 冲突数据集名称 (默认: "{workspace}_conflicts_{date}")
            label_question: 标签问题名称 (默认 "label")
            cot_question: COT问题名称 (默认 "COT")
            target_datasets: 指定要检测的数据集列表 (可选，默认检测所有)
            
        Returns:
            Dict 包含处理结果:
            {
                "conflict_dataset": 冲突数据集名称,
                "total_datasets_checked": 检查的数据集数量,
                "total_conflicts_found": 发现的冲突总数,
                "new_conflicts_added": 新添加的冲突数,
                "already_exists": 已存在的冲突数,
                "datasets_with_conflicts": 有冲突的数据集列表
            }
        """
        # 生成冲突数据集名称
        today = datetime.now().strftime("%Y%m%d")
        if not conflict_dataset_name:
            conflict_dataset_name = f"{workspace}_conflicts_{today}"
        
        result = {
            "conflict_dataset": conflict_dataset_name,
            "total_datasets_checked": 0,
            "total_conflicts_found": 0,
            "new_conflicts_added": 0,
            "already_exists": 0,
            "datasets_with_conflicts": []
        }
        
        print(f"\n{'='*60}")
        print(f"Conflict Detection for Workspace: {workspace}")
        print(f"Conflict Dataset: {conflict_dataset_name}")
        print(f"{'='*60}\n")
        
        try:
            # 获取workspace下所有数据集
            all_datasets = self.list_datasets(workspace=workspace)
            
            # 过滤掉冲突数据集本身
            datasets_to_check = [
                ds for ds in all_datasets 
                if not ds["name"].endswith("_conflicts") and "_conflicts_" not in ds["name"]
            ]
            
            # 如果指定了目标数据集，只检测这些
            if target_datasets:
                datasets_to_check = [
                    ds for ds in datasets_to_check 
                    if ds["name"] in target_datasets
                ]
            
            if not datasets_to_check:
                print(f"⚠ No datasets to check in workspace '{workspace}'")
                return result
            
            result["total_datasets_checked"] = len(datasets_to_check)
            
            # 收集所有冲突
            all_conflicts = []
            all_labels = set()
            
            for ds_info in datasets_to_check:
                dataset_name = ds_info["name"]
                print(f"\n[Checking] {dataset_name}...")
                
                # 获取数据集的labels（用于创建冲突数据集的问题选项）
                dataset = self.get_dataset(dataset_name, workspace=workspace)
                if dataset and dataset.settings and dataset.settings.questions:
                    for q in dataset.settings.questions:
                        if q.name == label_question and hasattr(q, 'labels') and q.labels:
                            all_labels.update(q.labels)
                
                # 检测冲突
                conflicts = self.detect_conflicts_for_dataset(
                    dataset_name=dataset_name,
                    label_question=label_question,
                    cot_question=cot_question,
                    workspace=workspace
                )
                
                if conflicts:
                    all_conflicts.extend(conflicts)
                    result["datasets_with_conflicts"].append(dataset_name)
            
            result["total_conflicts_found"] = len(all_conflicts)
            
            if not all_conflicts:
                print(f"\n✓ No conflicts found in any dataset")
                return result
            
            print(f"\n📊 Total conflicts found: {len(all_conflicts)}")
            
            # 检查冲突数据集是否存在
            conflict_dataset = self.get_dataset(conflict_dataset_name, workspace=workspace)
            
            if not conflict_dataset:
                # 创建新的冲突数据集
                labels_list = list(all_labels) if all_labels else ["positive", "negative", "neutral"]
                conflict_dataset = self._create_conflict_review_dataset(
                    conflict_dataset_name=conflict_dataset_name,
                    labels=labels_list,
                    workspace=workspace
                )
                
                if not conflict_dataset:
                    print(f"✗ Failed to create conflict dataset")
                    return result
            
            # 获取已存在的记录ID
            existing_record_ids = self._get_existing_conflict_record_ids(conflict_dataset)
            print(f"ℹ Existing records in conflict dataset: {len(existing_record_ids)}")
            
            # 准备新记录
            new_records = []
            
            for conflict in all_conflicts:
                original_record_id = conflict["record_id"]
                
                # 检查是否已存在
                if original_record_id in existing_record_ids:
                    result["already_exists"] += 1
                    continue
                
                # 格式化原始文本
                original_text = conflict["fields"].get("text", "")
                if not original_text:
                    # 尝试其他可能的字段名
                    for key in ["content", "input", "question", "query"]:
                        if key in conflict["fields"]:
                            original_text = conflict["fields"][key]
                            break
                
                # 格式化标注对比表格
                annotation_comparison = self._format_annotation_comparison_table(
                    conflict["annotator_1"], 
                    conflict["annotator_2"]
                )
                
                # 冲突信息（简化版）
                conflict_info = f"""**数据集**: {conflict["original_dataset"]}  
**记录ID**: {original_record_id}
"""
                
                # 构建记录
                record_data = {
                    "fields": {
                        "original_text": original_text,
                        "annotation_comparison": annotation_comparison,
                        "conflict_info": conflict_info,
                    },
                    "metadata": {
                        "original_dataset": conflict["original_dataset"],
                        "original_record_id": original_record_id,
                        "annotator_1_username": conflict["annotator_1"]["username"],
                        "annotator_2_username": conflict["annotator_2"]["username"],
                        "label_1": str(conflict["annotator_1"]["label"]),
                        "label_2": str(conflict["annotator_2"]["label"]),
                    }
                }
                
                new_records.append(record_data)
            
            # 添加新记录
            if new_records:
                try:
                    conflict_dataset.records.log(new_records)
                    result["new_conflicts_added"] = len(new_records)
                    print(f"✓ Added {len(new_records)} new conflict records")
                except Exception as e:
                    print(f"✗ Failed to add records: {e}")
            
            print(f"\n{'='*60}")
            print(f"✓ Conflict detection completed!")
            print(f"  - Datasets checked: {result['total_datasets_checked']}")
            print(f"  - Total conflicts found: {result['total_conflicts_found']}")
            print(f"  - New conflicts added: {result['new_conflicts_added']}")
            print(f"  - Already existed: {result['already_exists']}")
            print(f"  - Conflict dataset: {conflict_dataset_name}")
            print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            print(f"✗ Failed to create/update conflict dataset: {e}")
            import traceback
            traceback.print_exc()
            return result
    
    def get_conflict_statistics(
        self,
        conflict_dataset_name: str,
        workspace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取冲突数据集的统计信息
        
        Args:
            conflict_dataset_name: 冲突数据集名称
            workspace: 工作区名称 (可选)
            
        Returns:
            Dict 包含统计信息:
            {
                "total_conflicts": 总冲突数,
                "reviewed": 已审核数,
                "pending": 待审核数,
                "by_dataset": {数据集名: 冲突数},
                "by_annotator_pair": {标注者对: 冲突数},
                "label_conflict_matrix": {label1_vs_label2: 次数}
            }
        """
        dataset = self.get_dataset(conflict_dataset_name, workspace=workspace)
        if not dataset:
            print(f"✗ Conflict dataset '{conflict_dataset_name}' not found")
            return {}
        
        stats = {
            "total_conflicts": 0,
            "reviewed": 0,
            "pending": 0,
            "by_dataset": defaultdict(int),
            "by_annotator_pair": defaultdict(int),
            "label_conflict_matrix": defaultdict(int)
        }
        
        try:
            for record in dataset.records:
                stats["total_conflicts"] += 1
                
                # 检查是否已审核
                has_review = False
                if record.responses:
                    for resp in record.responses:
                        if resp.question_name == "correct_label" and resp.status == "submitted":
                            has_review = True
                            break
                
                if has_review:
                    stats["reviewed"] += 1
                else:
                    stats["pending"] += 1
                
                # 统计按数据集分布
                if record.metadata:
                    original_dataset = record.metadata.get("original_dataset", "unknown")
                    stats["by_dataset"][original_dataset] += 1
                    
                    # 统计标注者对
                    a1 = record.metadata.get("annotator_1_username", "?")
                    a2 = record.metadata.get("annotator_2_username", "?")
                    pair = f"{a1} vs {a2}"
                    stats["by_annotator_pair"][pair] += 1
                    
                    # 统计标签冲突矩阵
                    l1 = record.metadata.get("label_1", "?")
                    l2 = record.metadata.get("label_2", "?")
                    conflict_pair = f"{l1} vs {l2}"
                    stats["label_conflict_matrix"][conflict_pair] += 1
            
            # 转换为普通dict
            stats["by_dataset"] = dict(stats["by_dataset"])
            stats["by_annotator_pair"] = dict(stats["by_annotator_pair"])
            stats["label_conflict_matrix"] = dict(stats["label_conflict_matrix"])
            
            return stats
            
        except Exception as e:
            print(f"✗ Failed to get conflict statistics: {e}")
            # 确保返回普通dict而不是defaultdict，避免下游调用出错
            stats["by_dataset"] = dict(stats["by_dataset"])
            stats["by_annotator_pair"] = dict(stats["by_annotator_pair"])
            stats["label_conflict_matrix"] = dict(stats["label_conflict_matrix"])
            return stats

    def import_workspace_datasets(
        self,
        import_dir: str,
        target_workspace: str,
        overwrite_existing: bool = False
    ) -> bool:
        """
        从目录导入整个 workspace 的数据集（包括 settings、数据和标注）
        
        Args:
            import_dir: 导入目录（包含导出的数据集文件）
            target_workspace: 目标工作区名称
            overwrite_existing: 是否覆盖已存在的数据集
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 查找 workspace summary 文件
            summary_files = [f for f in os.listdir(import_dir) if f.startswith("workspace_summary_") and f.endswith(".json")]
            
            if not summary_files:
                print(f"⚠ No workspace summary file found in {import_dir}")
                print(f"  Will attempt to import all CSV/Parquet files in the directory...")
                
                # 没有 summary，尝试导入目录中的所有数据文件
                data_files = []
                for f in os.listdir(import_dir):
                    if (f.endswith(".csv") or f.endswith(".parquet")) and not f.endswith("_summary.json"):
                        data_files.append(f)
                
                if not data_files:
                    print(f"✗ No data files found in {import_dir}")
                    return False
                
                datasets_to_import = [{"file": f, "name": None} for f in data_files]
                
            else:
                # 读取 workspace summary
                summary_file = os.path.join(import_dir, sorted(summary_files)[-1])  # 使用最新的
                print(f"Found workspace summary: {summary_file}")
                
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                
                datasets_to_import = []
                
                # 根据 summary 中的数据集列表查找对应的数据文件
                for dataset_name in summary.get("datasets", []):
                    # 查找匹配的数据文件
                    dataset_safe_name = dataset_name.replace("/", "_").replace("\\", "_")
                    
                    # 查找数据文件
                    data_file = None
                    settings_file = None
                    
                    for f in os.listdir(import_dir):
                        if dataset_safe_name in f:
                            if f.endswith(".csv") or f.endswith(".parquet"):
                                if "_settings" not in f and "_summary" not in f:
                                    data_file = f
                            elif f.endswith("_settings.json"):
                                settings_file = f
                    
                    if data_file:
                        datasets_to_import.append({
                            "name": dataset_name,
                            "file": data_file,
                            "settings": settings_file
                        })
                    else:
                        print(f"⚠ Data file not found for dataset: {dataset_name}")
            
            if not datasets_to_import:
                print(f"✗ No datasets to import")
                return False
            
            print(f"\n{'='*60}")
            print(f"Importing workspace to '{target_workspace}' - {len(datasets_to_import)} datasets")
            print(f"Import directory: {import_dir}")
            print(f"{'='*60}\n")
            
            # 导入每个数据集
            success_count = 0
            failed_datasets = []
            
            for i, dataset_info in enumerate(datasets_to_import, 1):
                dataset_name = dataset_info.get("name")
                data_file = dataset_info["file"]
                settings_file = dataset_info.get("settings")
                
                print(f"\n[{i}/{len(datasets_to_import)}] Importing: {dataset_name or data_file}")
                
                # 检查数据集是否已存在
                existing_dataset = self.get_dataset(dataset_name, workspace=target_workspace) if dataset_name else None
                
                if existing_dataset:
                    if overwrite_existing:
                        print(f"  ⚠ Dataset '{dataset_name}' already exists. Deleting...")
                        existing_dataset.delete()
                    else:
                        print(f"  ⚠ Dataset '{dataset_name}' already exists. Skipping (use overwrite_existing=True to replace)")
                        failed_datasets.append(dataset_name or data_file)
                        continue
                
                # 导入数据集
                import_file_path = os.path.join(import_dir, data_file)
                settings_file_path = os.path.join(import_dir, settings_file) if settings_file else None
                
                success = self.import_dataset_with_annotations(
                    import_file=import_file_path,
                    dataset_name=dataset_name,
                    workspace=target_workspace,
                    create_if_not_exists=True,
                    settings_file=settings_file_path
                )
                
                if success:
                    success_count += 1
                else:
                    failed_datasets.append(dataset_name or data_file)
            
            print(f"\n{'='*60}")
            print(f"✓ Workspace import completed!")
            print(f"  - Successfully imported: {success_count}/{len(datasets_to_import)} datasets")
            if failed_datasets:
                print(f"  - Failed: {', '.join(failed_datasets)}")
            print(f"{'='*60}\n")
            
            return success_count > 0
            
        except Exception as e:
            print(f"✗ Failed to import workspace datasets: {e}")
            import traceback
            traceback.print_exc()
            return False


# Convenience functions for quick operations
def create_simple_text_classification_dataset(
    name: str,
    labels: List[str],
    workspace: Optional[str] = None,
    overwrite: bool = False
) -> Optional[rg.Dataset]:
    """
    Quick function to create a simple text classification dataset
    
    Args:
        name: Dataset name
        labels: List of classification labels
        workspace: Workspace name (optional)
        overwrite: Overwrite if exists
        
    Returns:
        Created Dataset object
    """
    manager = ArgillaDatasetManager()
    
    # Create simple settings
    text_field = rg.TextField(name="text", title="Text")
    label_question = rg.LabelQuestion(
        name="label",
        title="Label",
        labels=labels
    )
    settings = rg.Settings(
        fields=[text_field],
        questions=[label_question]
    )
    
    try:
        return manager.create_dataset(name, settings, workspace, overwrite)
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def delete_dataset(name: str, workspace: Optional[str] = None) -> bool:
    """
    Quick function to delete a dataset
    
    Args:
        name: Dataset name
        workspace: Workspace name (optional, 避免误删其他workspace的同名数据集)
    """
    manager = ArgillaDatasetManager()
    return manager.delete_dataset(name, workspace=workspace)


def list_all_datasets() -> List[Dict[str, Any]]:
    """Quick function to list all datasets"""
    manager = ArgillaDatasetManager()
    return manager.list_datasets()


def export_dataset_to_disk(dataset_name: str, export_path: str, workspace: Optional[str] = None) -> bool:
    """
    Quick function to export dataset
    
    Args:
        dataset_name: Dataset name
        export_path: Export path
        workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
    """
    manager = ArgillaDatasetManager()
    return manager.export_dataset(dataset_name, export_path, workspace=workspace)


def import_dataset_from_disk(import_path: str, new_name: Optional[str] = None) -> Optional[rg.Dataset]:
    """Quick function to import dataset"""
    manager = ArgillaDatasetManager()
    return manager.import_dataset(import_path, new_name)


def add_records_with_responses(
    dataset_name: str,
    records: List[Dict[str, Any]],
    user_id: Optional[str] = None,
    workspace: Optional[str] = None
) -> bool:
    """
    Quick function to add records with submitted responses
    
    Args:
        dataset_name: 数据集名称
        records: 记录列表 (包含 fields, responses, metadata, suggestions)
        user_id: 用户ID (可选)
        workspace: 工作区名称 (可选)
    """
    manager = ArgillaDatasetManager()
    return manager.add_records_with_responses(dataset_name, records, user_id, workspace)


def export_dataset_annotations(
    dataset_name: str,
    export_dir: str,
    format: str = "csv",
    workspace: Optional[str] = None
) -> bool:
    """
    Quick function to export dataset with all annotations
    
    Args:
        dataset_name: 数据集名称
        export_dir: 导出目录
        format: 导出格式 ('csv' 或 'parquet')
        workspace: 工作区名称 (可选)
    """
    manager = ArgillaDatasetManager()
    return manager.export_dataset_with_annotations(dataset_name, export_dir, format, workspace)


def export_workspace(
    workspace: str,
    export_base_dir: str,
    format: str = "csv"
) -> bool:
    """
    Quick function to export entire workspace (all datasets)
    
    Args:
        workspace: 工作区名称
        export_base_dir: 导出根目录
        format: 导出格式 ('csv' 或 'parquet')
    """
    manager = ArgillaDatasetManager()
    return manager.export_workspace_datasets(workspace, export_base_dir, format)


def import_dataset_annotations(
    import_file: str,
    dataset_name: Optional[str] = None,
    workspace: Optional[str] = None
) -> bool:
    """
    Quick function to import dataset with annotations from CSV/Parquet
    
    Args:
        import_file: 导入文件路径
        dataset_name: 目标数据集名称 (可选)
        workspace: 工作区名称 (可选)
    """
    manager = ArgillaDatasetManager()
    return manager.import_dataset_with_annotations(import_file, dataset_name, workspace)


def detect_and_create_conflict_dataset(
    workspace: str,
    conflict_dataset_name: Optional[str] = None,
    label_question: str = "label",
    cot_question: str = "COT",
    target_datasets: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    检测workspace下所有数据集的标注冲突，并创建/更新冲突审核数据集
    
    这是一个便捷函数，用于每天分析标注冲突并生成冲突数据集供第三方审核。
    
    功能说明:
    1. 遍历指定workspace下的所有数据集（或指定的数据集列表）
    2. 检测每个数据集中两位标注者的label是否一致
    3. 如果label不一致，则视为冲突
    4. 将冲突记录添加到冲突审核数据集中
    5. 如果记录已存在于冲突数据集中，则跳过
    
    冲突数据集的结构:
    - 字段:
        - original_text: 原始文本
        - annotation_comparison: 标注对比HTML表格（表头为标注者姓名，行为Label/COT/时间）
        - conflict_info: 冲突信息（数据集名称、记录ID）
    - 元数据:
        - original_dataset: 原始数据集名称
        - original_record_id: 原始记录ID
        - annotator_1_username: 标注者1用户名
        - annotator_2_username: 标注者2用户名
        - label_1: 标注者1的标签
        - label_2: 标注者2的标签
    - 问题:
        - correct_label: 审核者选择的正确标签
        - review_reason: 审核理由（可选）
    
    Args:
        workspace: 工作区名称
        conflict_dataset_name: 冲突数据集名称 (默认: "{workspace}_conflicts_{date}")
        label_question: 标签问题名称 (默认 "label")
        cot_question: COT问题名称 (默认 "COT")
        target_datasets: 指定要检测的数据集列表 (可选，默认检测所有)
        
    Returns:
        Dict 包含处理结果:
        {
            "conflict_dataset": 冲突数据集名称,
            "total_datasets_checked": 检查的数据集数量,
            "total_conflicts_found": 发现的冲突总数,
            "new_conflicts_added": 新添加的冲突数,
            "already_exists": 已存在的冲突数,
            "datasets_with_conflicts": 有冲突的数据集列表
        }
        
    Example:
        # 检测整个workspace的冲突
        result = detect_and_create_conflict_dataset("my_workspace")
        print(f"发现 {result['total_conflicts_found']} 个冲突")
        print(f"新添加 {result['new_conflicts_added']} 个冲突记录")
        
        # 只检测指定数据集
        result = detect_and_create_conflict_dataset(
            workspace="my_workspace",
            target_datasets=["dataset1", "dataset2"],
            label_question="intent",  # 自定义标签问题名称
            cot_question="reason"  # 自定义COT问题名称
        )
    """
    manager = ArgillaDatasetManager()
    return manager.create_or_update_conflict_dataset(
        workspace=workspace,
        conflict_dataset_name=conflict_dataset_name,
        label_question=label_question,
        cot_question=cot_question,
        target_datasets=target_datasets
    )


def get_conflict_statistics(
    conflict_dataset_name: str,
    workspace: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取冲突数据集的统计信息
    
    Args:
        conflict_dataset_name: 冲突数据集名称
        workspace: 工作区名称 (可选)
        
    Returns:
        Dict 包含统计信息
    """
    manager = ArgillaDatasetManager()
    return manager.get_conflict_statistics(conflict_dataset_name, workspace)


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ArgillaDatasetManager()
    
    # List all datasets
    print("All datasets:")
    datasets = manager.list_datasets()
    for ds in datasets:
        print(f"  - {ds['name']} (Workspace: {ds['workspace']})")
    
    # Example: Get dataset info
    # info = manager.get_dataset_info("my_dataset")
    # if info:
    #     print(f"\nDataset Info: {info}")
    
    # Example: Create a simple dataset
    # dataset = create_simple_text_classification_dataset(
    #     name="test_dataset",
    #     labels=["positive", "negative", "neutral"]
    # )
    
    # Example: Export dataset (legacy - without annotations)
    # manager.export_dataset("my_dataset", "./exports/my_dataset")
    
    # Example: Export dataset with annotations (NEW!)
    # manager.export_dataset_with_annotations(
    #     dataset_name="my_dataset",
    #     export_dir="./exports/annotations",
    #     format="csv",  # or "parquet"
    #     workspace="my_workspace"
    # )
    
    # Example: Export entire workspace (NEW!)
    # manager.export_workspace_datasets(
    #     workspace="my_workspace",
    #     export_base_dir="./exports",
    #     format="csv"
    # )
    
    # Example: Import dataset with annotations (NEW!)
    # manager.import_dataset_with_annotations(
    #     import_file="./exports/my_workspace/2025-11-28/my_dataset_20251128.csv",
    #     workspace="my_workspace"
    # )
    
    # Example: Delete dataset
    # manager.delete_dataset("test_dataset")
    
    # ========================================================================
    # 冲突检测与审核数据集生成 (Conflict Detection and Review Dataset)
    # ========================================================================
    
    # 示例1: 检测整个workspace的标注冲突，生成冲突审核数据集
    # ---------------------------------------------------------------
    # 这个方法会:
    # 1. 遍历workspace下所有数据集
    # 2. 检测每个数据集中两位标注者的label是否冲突
    # 3. 如果冲突，生成冲突记录到审核数据集
    # 4. 如果记录已存在，跳过
    #
    # result = detect_and_create_conflict_dataset(
    #     workspace="my_workspace",  # 要检测的workspace名称
    #     # conflict_dataset_name="custom_conflicts",  # 可选：自定义冲突数据集名称
    #     # label_question="label",  # 标签问题名称，默认 "label"
    #     # cot_question="COT",  # COT问题名称，默认 "COT"
    # )
    # print(f"检测结果: {result}")
    
    # 示例2: 只检测指定的数据集
    # ---------------------------------------------------------------
    # result = detect_and_create_conflict_dataset(
    #     workspace="my_workspace",
    #     target_datasets=["dataset1", "dataset2"],  # 只检测这两个数据集
    #     label_question="intent",  # 如果你的标签问题名不是 "label"
    #     cot_question="reason",  # 如果你的COT问题名不是 "COT"
    # )
    
    # 示例3: 获取冲突数据集的统计信息
    # ---------------------------------------------------------------
    # stats = get_conflict_statistics(
    #     conflict_dataset_name="my_workspace_conflicts_20241204",
    #     workspace="my_workspace"
    # )
    # print(f"冲突统计:")
    # print(f"  - 总冲突数: {stats.get('total_conflicts', 0)}")
    # print(f"  - 已审核: {stats.get('reviewed', 0)}")
    # print(f"  - 待审核: {stats.get('pending', 0)}")
    # print(f"  - 按数据集分布: {stats.get('by_dataset', {})}")
    # print(f"  - 标签冲突矩阵: {stats.get('label_conflict_matrix', {})}")
    
    # 示例4: 使用 ArgillaDatasetManager 类直接调用
    # ---------------------------------------------------------------
    # manager = ArgillaDatasetManager()
    # 
    # # 检测单个数据集的冲突
    # conflicts = manager.detect_conflicts_for_dataset(
    #     dataset_name="my_dataset",
    #     label_question="label",
    #     cot_question="COT",
    #     workspace="my_workspace"
    # )
    # print(f"发现 {len(conflicts)} 个冲突")
    # for conflict in conflicts:
    #     print(f"  - 记录ID: {conflict['record_id']}")
    #     print(f"    标注者1: {conflict['annotator_1']['username']} -> {conflict['annotator_1']['label']}")
    #     print(f"    标注者2: {conflict['annotator_2']['username']} -> {conflict['annotator_2']['label']}")
    #
    # # 创建/更新冲突数据集
    # result = manager.create_or_update_conflict_dataset(
    #     workspace="my_workspace",
    #     conflict_dataset_name="review_conflicts",
    #     label_question="label",
    #     cot_question="COT"
    # )
    
    # 示例5: 定时任务使用 (每日冲突检测)
    # ---------------------------------------------------------------
    # 可以将以下代码放入定时任务中，每天执行一次
    #
    # import schedule
    # import time
    # 
    # def daily_conflict_check():
    #     """每日冲突检测任务"""
    #     print(f"[{datetime.now()}] 开始每日冲突检测...")
    #     
    #     workspaces_to_check = ["workspace1", "workspace2"]
    #     
    #     for workspace in workspaces_to_check:
    #         result = detect_and_create_conflict_dataset(workspace=workspace)
    #         print(f"  [{workspace}] 发现 {result['total_conflicts_found']} 个冲突, "
    #               f"新增 {result['new_conflicts_added']} 条")
    #     
    #     print(f"[{datetime.now()}] 冲突检测完成")
    # 
    # # 每天早上9点执行
    # schedule.every().day.at("09:00").do(daily_conflict_check)
    # 
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)

