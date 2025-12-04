"""
Argilla Common Utilities

This module provides utility functions for common Argilla operations that don't fit
into user or dataset categories, including:
- Workspace management
- Record operations and queries
- Response and annotation management
- Statistics and analytics
- Conflict detection and resolution
- Connection testing and validation

IMPORTANT FIXES:
1. 所有记录/冲突检测方法都支持 workspace 参数，避免误操作其他workspace的同名数据集
2. 修复了 Response API 访问：使用 resp.value + resp.question_name 而非 resp.values[question]

建议使用时明确指定 workspace，例如:
    conflicts = detect_annotation_conflicts("dataset", "label", workspace="my_workspace")

Author: Argilla Utils
Version: 1.1.0 (2025-11-25: 添加workspace支持，修复Response API)
"""

import os
from typing import Optional, List, Dict, Any, Callable
from collections import Counter, defaultdict
from dotenv import load_dotenv
import argilla as rg
from argilla._exceptions import ArgillaError


class ArgillaWorkspaceManager:
    """Manager class for Argilla workspace operations"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Argilla Workspace Manager
        
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
    
    def create_workspace(self, name: str) -> Optional[rg.Workspace]:
        """
        Create a new workspace
        
        Args:
            name: Workspace name
            
        Returns:
            Created Workspace object, or None if failed
        """
        # Check if workspace exists
        existing = self.get_workspace(name)
        if existing:
            print(f"ℹ Workspace '{name}' already exists")
            return existing
        
        try:
            workspace = rg.Workspace(name=name)
            workspace.create()
            print(f"✓ Workspace '{name}' created successfully")
            return workspace
        except Exception as e:
            print(f"✗ Failed to create workspace '{name}': {e}")
            return None
    
    def delete_workspace(self, name: str) -> bool:
        """
        Delete a workspace
        
        Args:
            name: Workspace name
            
        Returns:
            True if successful, False otherwise
        """
        workspace = self.get_workspace(name)
        if not workspace:
            print(f"✗ Workspace '{name}' not found")
            return False
        
        try:
            workspace.delete()
            print(f"✓ Workspace '{name}' deleted successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to delete workspace '{name}': {e}")
            return False
    
    def get_workspace(self, name: str) -> Optional[rg.Workspace]:
        """
        Get workspace by name
        
        Args:
            name: Workspace name
            
        Returns:
            Workspace object if found, None otherwise
        """
        try:
            workspace = self.client.workspaces(name)
            return workspace
        except Exception:
            return None
    
    def list_workspaces(self) -> List[Dict[str, Any]]:
        """
        List all workspaces
        
        Returns:
            List of dictionaries containing workspace information
        """
        try:
            workspaces = []
            for ws in self.client.workspaces:
                workspaces.append({
                    "id": str(ws.id),
                    "name": ws.name,
                })
            return workspaces
        except Exception as e:
            print(f"✗ Failed to list workspaces: {e}")
            return []
    
    def list_workspace_users(self, workspace_name: str) -> List[Dict[str, Any]]:
        """
        List all users in a workspace
        
        Args:
            workspace_name: Workspace name
            
        Returns:
            List of user information dictionaries
        """
        workspace = self.get_workspace(workspace_name)
        if not workspace:
            print(f"✗ Workspace '{workspace_name}' not found")
            return []
        
        try:
            users = []
            for user in workspace.users:
                users.append({
                    "id": str(user.id),
                    "username": user.username,
                    "role": user.role if hasattr(user, 'role') else None,
                })
            return users
        except Exception as e:
            print(f"✗ Failed to list workspace users: {e}")
            return []


class ArgillaRecordManager:
    """Manager class for Argilla record operations"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Argilla Record Manager
        
        Args:
            api_url: Argilla API URL
            api_key: Argilla API Key
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
    
    def query_records(
        self,
        dataset_name: str,
        filter_func: Optional[Callable] = None,
        limit: Optional[int] = None,
        workspace: Optional[str] = None
    ) -> List[Any]:
        """
        Query and filter records from a dataset
        
        Args:
            dataset_name: Dataset name
            filter_func: Function to filter records (takes record, returns bool)
            limit: Maximum number of records to return
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            List of records matching the filter
        """
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return []
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return []
        
        try:
            records = []
            count = 0
            
            for record in dataset.records:
                if filter_func is None or filter_func(record):
                    records.append(record)
                    count += 1
                    
                    if limit and count >= limit:
                        break
            
            return records
        except Exception as e:
            print(f"✗ Failed to query records: {e}")
            return []
    
    def get_records_by_metadata(
        self,
        dataset_name: str,
        metadata_filters: Dict[str, Any],
        workspace: Optional[str] = None
    ) -> List[Any]:
        """
        Get records matching specific metadata values
        
        Args:
            dataset_name: Dataset name
            metadata_filters: Dictionary of metadata key-value pairs to match
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            List of matching records
        """
        def filter_func(record):
            if not record.metadata:
                return False
            
            for key, value in metadata_filters.items():
                if key not in record.metadata or record.metadata[key] != value:
                    return False
            return True
        
        return self.query_records(dataset_name, filter_func, workspace=workspace)
    
    def get_pending_records(self, dataset_name: str, workspace: Optional[str] = None) -> List[Any]:
        """
        Get all records without any responses
        
        Args:
            dataset_name: Dataset name
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            List of pending records
        """
        def filter_func(record):
            return not record.responses or len(record.responses) == 0
        
        return self.query_records(dataset_name, filter_func, workspace=workspace)
    
    def get_submitted_records(self, dataset_name: str, workspace: Optional[str] = None) -> List[Any]:
        """
        Get all records with at least one submitted response
        
        Args:
            dataset_name: Dataset name
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            List of submitted records
        """
        def filter_func(record):
            if not record.responses:
                return False
            return any(r.status == "submitted" for r in record.responses)
        
        return self.query_records(dataset_name, filter_func, workspace=workspace)
    
    def detect_conflicts(
        self,
        dataset_name: str,
        question_name: str,
        min_responses: int = 2,
        workspace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect records with conflicting annotations
        
        Args:
            dataset_name: Dataset name
            question_name: Question name to check for conflicts
            min_responses: Minimum number of responses required to check for conflicts
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            List of conflict information dictionaries
        """
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return []
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return []
        
        conflicts = []
        
        try:
            for record in dataset.records:
                if not record.responses or len(record.responses) < min_responses:
                    continue
                
                # Extract submitted responses for the question
                user_answers = {}
                for resp in record.responses:
                    if resp.status != "submitted":
                        continue
                    
                    # Argilla 2.x: Response对象通过question_name访问，value是实际值
                    try:
                        # 尝试获取该question的响应值
                        val = getattr(resp, 'value', None)
                        if val is not None and hasattr(resp, 'question_name') and resp.question_name == question_name:
                            user_answers[resp.user_id] = val
                    except (AttributeError, KeyError):
                        continue
                
                # Check for conflicts
                if len(user_answers) >= min_responses:
                    unique_answers = set(user_answers.values())
                    
                    if len(unique_answers) > 1:
                        conflicts.append({
                            "record_id": str(record.id),
                            "text": record.fields.get("text", ""),
                            "responses": user_answers,
                            "unique_answers": list(unique_answers),
                            "metadata": record.metadata
                        })
            
            print(f"✓ Found {len(conflicts)} conflicts in dataset '{dataset_name}'")
            return conflicts
            
        except Exception as e:
            print(f"✗ Failed to detect conflicts: {e}")
            return []
    
    def get_annotation_stats(
        self,
        dataset_name: str,
        question_name: str,
        workspace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get annotation statistics for a specific question
        
        Args:
            dataset_name: Dataset name
            question_name: Question name
            workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
            
        Returns:
            Dictionary with annotation statistics
        """
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return {}
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return {}
        
        try:
            total_records = 0
            records_with_responses = 0
            all_answers = []
            user_contributions = defaultdict(int)
            
            for record in dataset.records:
                total_records += 1
                
                if not record.responses:
                    continue
                
                has_response = False
                for resp in record.responses:
                    if resp.status != "submitted":
                        continue
                    
                    # Argilla 2.x: Response对象通过question_name访问，value是实际值
                    try:
                        val = getattr(resp, 'value', None)
                        if val is not None and hasattr(resp, 'question_name') and resp.question_name == question_name:
                            all_answers.append(val)
                            user_contributions[resp.user_id] += 1
                            has_response = True
                    except (AttributeError, KeyError):
                        continue
                
                if has_response:
                    records_with_responses += 1
            
            # Calculate statistics
            answer_distribution = Counter(all_answers)
            
            stats = {
                "total_records": total_records,
                "annotated_records": records_with_responses,
                "pending_records": total_records - records_with_responses,
                "completion_rate": records_with_responses / total_records if total_records > 0 else 0,
                "total_annotations": len(all_answers),
                "answer_distribution": dict(answer_distribution),
                "user_contributions": dict(user_contributions),
            }
            
            return stats
            
        except Exception as e:
            print(f"✗ Failed to get annotation stats: {e}")
            return {}
    
    def get_annotator_progress(
        self,
        dataset_name: str,
        workspace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取每个标注员的标注进度和统计信息
        
        Args:
            dataset_name: 数据集名称
            workspace: 工作区名称 (可选)
            
        Returns:
            Dict 包含每个标注员的详细信息:
            {
                "total_records": 总记录数,
                "annotators": {
                    "user_id": {
                        "username": 用户名,
                        "submitted_count": 已提交数量,
                        "draft_count": 草稿数量,
                        "total_annotations": 总标注数,
                        "completion_rate": 完成率,
                        "answer_distribution": 答案分布,
                        "annotated_records": [记录ID列表]
                    }
                }
            }
        """
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return {}
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return {}
        
        try:
            total_records = 0
            annotators = defaultdict(lambda: {
                "username": None,
                "submitted_count": 0,
                "draft_count": 0,
                "discarded_count": 0,
                "total_annotations": 0,
                "answer_distribution": defaultdict(int),
                "annotated_records": [],
                "question_stats": defaultdict(lambda: {"submitted": 0, "draft": 0, "answers": []})
            })
            
            # 获取用户信息映射
            user_mapping = {}
            try:
                for user in self.client.users:
                    user_mapping[str(user.id)] = user.username
            except:
                pass
            
            # 遍历所有记录
            for record in dataset.records:
                total_records += 1
                
                if not record.responses:
                    continue
                
                # 统计每个用户的响应
                for resp in record.responses:
                    user_id = str(resp.user_id)
                    question_name = resp.question_name
                    
                    # 设置用户名
                    if annotators[user_id]["username"] is None:
                        annotators[user_id]["username"] = user_mapping.get(user_id, user_id)
                    
                    # 统计状态
                    if resp.status == "submitted":
                        annotators[user_id]["submitted_count"] += 1
                        annotators[user_id]["question_stats"][question_name]["submitted"] += 1
                        annotators[user_id]["question_stats"][question_name]["answers"].append(resp.value)
                        
                        # 记录答案分布
                        if resp.value:
                            annotators[user_id]["answer_distribution"][str(resp.value)] += 1
                        
                        # 记录已标注的记录ID
                        if str(record.id) not in annotators[user_id]["annotated_records"]:
                            annotators[user_id]["annotated_records"].append(str(record.id))
                    
                    elif resp.status == "draft":
                        annotators[user_id]["draft_count"] += 1
                        annotators[user_id]["question_stats"][question_name]["draft"] += 1
                    
                    elif resp.status == "discarded":
                        annotators[user_id]["discarded_count"] += 1
                    
                    annotators[user_id]["total_annotations"] += 1
            
            # 计算完成率
            result = {
                "total_records": total_records,
                "annotators": {}
            }
            
            for user_id, stats in annotators.items():
                stats["completion_rate"] = stats["submitted_count"] / total_records if total_records > 0 else 0
                stats["answer_distribution"] = dict(stats["answer_distribution"])
                stats["question_stats"] = dict(stats["question_stats"])
                result["annotators"][user_id] = dict(stats)
            
            return result
            
        except Exception as e:
            print(f"✗ Failed to get annotator progress: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_user_annotation_detail(
        self,
        dataset_name: str,
        user_id: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取指定用户的详细标注记录
        
        Args:
            dataset_name: 数据集名称
            user_id: 用户ID (可选，默认为当前用户)
            workspace: 工作区名称 (可选)
            
        Returns:
            List[Dict] 包含用户标注的每条记录详情:
            [
                {
                    "record_id": 记录ID,
                    "fields": 记录字段内容,
                    "responses": 用户的响应列表,
                    "status": 响应状态,
                    "metadata": 记录元数据
                }
            ]
        """
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return []
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return []
        
        # 获取用户ID
        if user_id is None:
            user_id = str(self.client.me.id)
        else:
            user_id = str(user_id)
        
        try:
            user_records = []
            
            for record in dataset.records:
                if not record.responses:
                    continue
                
                # 查找该用户的响应
                user_responses = []
                for resp in record.responses:
                    if str(resp.user_id) == user_id:
                        user_responses.append({
                            "question_name": resp.question_name,
                            "value": resp.value,
                            "status": resp.status
                        })
                
                if user_responses:
                    user_records.append({
                        "record_id": str(record.id),
                        "fields": dict(record.fields) if record.fields else {},
                        "responses": user_responses,
                        "metadata": dict(record.metadata) if record.metadata else {}
                    })
            
            return user_records
            
        except Exception as e:
            print(f"✗ Failed to get user annotation detail: {e}")
            return []
    
    def compare_annotators(
        self,
        dataset_name: str,
        question_name: str,
        workspace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        对比不同标注员在同一记录上的标注结果，找出一致性和差异
        
        Args:
            dataset_name: 数据集名称
            question_name: 问题名称
            workspace: 工作区名称 (可选)
            
        Returns:
            Dict 包含:
            {
                "agreement_rate": 一致率,
                "conflicts": 冲突记录列表,
                "consensus": 有共识的记录列表,
                "annotator_comparison": 标注员间对比
            }
        """
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return {}
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return {}
        
        try:
            conflicts = []
            consensus = []
            multi_annotated_count = 0
            agreement_count = 0
            
            # 获取用户名映射
            user_mapping = {}
            try:
                for user in self.client.users:
                    user_mapping[str(user.id)] = user.username
            except:
                pass
            
            for record in dataset.records:
                if not record.responses:
                    continue
                
                # 收集该问题的所有已提交响应
                user_answers = {}
                for resp in record.responses:
                    if resp.status == "submitted" and resp.question_name == question_name:
                        user_id = str(resp.user_id)
                        username = user_mapping.get(user_id, user_id)
                        user_answers[username] = resp.value
                
                # 至少2个人标注才进行对比
                if len(user_answers) >= 2:
                    multi_annotated_count += 1
                    unique_answers = set(user_answers.values())
                    
                    record_info = {
                        "record_id": str(record.id),
                        "text": record.fields.get("text", "")[:100],  # 显示前100字符
                        "annotators": user_answers,
                        "unique_answers": list(unique_answers)
                    }
                    
                    if len(unique_answers) == 1:
                        # 所有人意见一致
                        agreement_count += 1
                        consensus.append(record_info)
                    else:
                        # 存在分歧
                        conflicts.append(record_info)
            
            # 计算一致率
            agreement_rate = agreement_count / multi_annotated_count if multi_annotated_count > 0 else 0
            
            result = {
                "total_multi_annotated": multi_annotated_count,
                "agreement_count": agreement_count,
                "conflict_count": len(conflicts),
                "agreement_rate": agreement_rate,
                "conflicts": conflicts,
                "consensus": consensus
            }
            
            return result
            
        except Exception as e:
            print(f"✗ Failed to compare annotators: {e}")
            return {}
    
    def update_record(
        self,
        dataset_name: str,
        record_id: str,
        fields: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        suggestions: Optional[Dict[str, Any]] = None,
        workspace: Optional[str] = None
    ) -> bool:
        """
        更新指定记录的内容
        
        Args:
            dataset_name: 数据集名称
            record_id: 记录ID
            fields: 要更新的字段内容 (dict: {field_name: new_value})
            metadata: 要更新的元数据 (dict: {metadata_key: new_value})
            suggestions: 要更新的建议标注 (dict: {question_name: suggestion_value})
            workspace: 工作区名称 (可选，避免误操作其他workspace的同名数据集)
            
        Returns:
            True if successful, False otherwise
            
        Example:
            # 更新记录的文本字段和元数据
            manager.update_record(
                "my_dataset",
                "record_123",
                fields={"text": "新的文本内容"},
                metadata={"category": "updated", "confidence": 0.95}
            )
            
            # 更新建议标注
            manager.update_record(
                "my_dataset",
                "record_123",
                suggestions={"label": "positive"}
            )
        """
        # 获取数据集
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return False
        
        try:
            # 查找目标记录
            target_record = None
            for record in dataset.records:
                if str(record.id) == str(record_id):
                    target_record = record
                    break
            
            if not target_record:
                print(f"✗ Record with ID '{record_id}' not found in dataset '{dataset_name}'")
                return False
            
            # 更新字段 (fields)
            if fields:
                for field_name, field_value in fields.items():
                    if field_name in target_record.fields:
                        target_record.fields[field_name] = field_value
                    else:
                        print(f"⚠ Warning: Field '{field_name}' not found in record, skipping")
            
            # 更新元数据 (metadata)
            if metadata:
                if target_record.metadata is None:
                    target_record.metadata = {}
                for meta_key, meta_value in metadata.items():
                    target_record.metadata[meta_key] = meta_value
            
            # 更新建议标注 (suggestions)
            if suggestions:
                for question_name, suggestion_value in suggestions.items():
                    # 检查question是否存在
                    question_exists = any(
                        q.name == question_name 
                        for q in dataset.settings.questions
                    )
                    
                    if question_exists:
                        # 创建或更新建议
                        if hasattr(target_record, 'suggestions') and target_record.suggestions:
                            # 更新现有建议
                            if question_name in target_record.suggestions:
                                target_record.suggestions[question_name].value = suggestion_value
                            else:
                                # 添加新建议
                                target_record.suggestions[question_name] = rg.Suggestion(
                                    question_name=question_name,
                                    value=suggestion_value
                                )
                        else:
                            # 初始化建议
                            target_record.suggestions = {
                                question_name: rg.Suggestion(
                                    question_name=question_name,
                                    value=suggestion_value
                                )
                            }
                    else:
                        print(f"⚠ Warning: Question '{question_name}' not found in dataset, skipping suggestion")
            
            # 保存更新到服务器
            target_record.update()
            
            print(f"✓ Record '{record_id}' updated successfully in dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            print(f"✗ Failed to update record: {e}")
            return False
    
    def submit_response(
        self,
        dataset_name: str,
        record_id: str,
        responses: Dict[str, Any],
        user_id: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> bool:
        """
        为记录添加或更新 Response，并设置状态为 submitted
        
        Args:
            dataset_name: 数据集名称
            record_id: 记录ID
            responses: Response内容 (dict: {question_name: answer_value})
            user_id: 用户ID (可选，默认为当前用户)
            workspace: 工作区名称 (可选，避免误操作其他workspace的同名数据集)
            
        Returns:
            True if successful, False otherwise
            
        Example:
            # 提交单个问题的答案
            manager.submit_response(
                "my_dataset",
                "record_123",
                {"label": "positive"}
            )
            
            # 提交多个问题的答案
            manager.submit_response(
                "my_dataset",
                "record_123",
                {"label": "positive", "sentiment": "good"}
            )
        """
        # 获取数据集
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return False
        
        try:
            # 查找目标记录
            target_record = None
            for record in dataset.records:
                if str(record.id) == str(record_id):
                    target_record = record
                    break
            
            if not target_record:
                print(f"✗ Record with ID '{record_id}' not found in dataset '{dataset_name}'")
                return False
            
            # 获取当前用户ID（如果未指定）
            if user_id is None:
                user_id = self.client.me.id
            
            # 为每个问题创建或更新 Response，状态设置为 submitted
            for question_name, answer_value in responses.items():
                # 检查 question 是否存在
                question = None
                for q in dataset.settings.questions:
                    if q.name == question_name:
                        question = q
                        break
                
                if not question:
                    print(f"⚠ Warning: Question '{question_name}' not found in dataset, skipping")
                    continue
                
                # 检查是否已有该用户对该问题的 response
                existing_response = None
                if target_record.responses:
                    for resp in target_record.responses:
                        if resp.question_name == question_name and str(resp.user_id) == str(user_id):
                            existing_response = resp
                            break
                
                if existing_response:
                    # 更新现有 response
                    existing_response.value = answer_value
                    existing_response.status = "submitted"  # 设置为 submitted 状态
                else:
                    # 创建新 response，状态为 submitted
                    new_response = rg.Response(
                        question_name=question_name,
                        value=answer_value,
                        user_id=user_id,
                        status="submitted"  # 关键：设置状态为 submitted
                    )
                    # 添加到记录
                    if not target_record.responses:
                        target_record.responses = []
                    target_record.responses.append(new_response)
            
            # 保存更新到服务器
            target_record.update()
            
            print(f"✓ Response submitted successfully for record '{record_id}' in dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            print(f"✗ Failed to submit response: {e}")
            return False
    
    def batch_submit_responses(
        self,
        dataset_name: str,
        records_data: List[Dict[str, Any]],
        workspace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        批量为记录提交 Response（状态为submitted）
        
        Args:
            dataset_name: 数据集名称
            records_data: 记录数据列表，每个元素包含:
                - record_id: 记录ID
                - responses: Response内容 {question_name: answer_value}
                - user_id: (可选) 用户ID
            workspace: 工作区名称 (可选，避免误操作其他workspace的同名数据集)
            
        Returns:
            Dict with success count and failed records
            
        Example:
            records = [
                {
                    "record_id": "rec_1",
                    "responses": {"label": "positive"}
                },
                {
                    "record_id": "rec_2",
                    "responses": {"label": "negative"},
                    "user_id": "user_123"
                }
            ]
            result = manager.batch_submit_responses("my_dataset", records)
        """
        success_count = 0
        failed_records = []
        
        for record_data in records_data:
            record_id = record_data.get("record_id")
            responses = record_data.get("responses", {})
            user_id = record_data.get("user_id")
            
            if not record_id or not responses:
                failed_records.append({
                    "record_id": record_id,
                    "error": "Missing record_id or responses"
                })
                continue
            
            success = self.submit_response(
                dataset_name,
                record_id,
                responses,
                user_id,
                workspace
            )
            
            if success:
                success_count += 1
            else:
                failed_records.append({
                    "record_id": record_id,
                    "error": "Submission failed"
                })
        
        result = {
            "total": len(records_data),
            "success": success_count,
            "failed": len(failed_records),
            "failed_records": failed_records
        }
        
        print(f"\n✓ Batch submission completed: {success_count}/{len(records_data)} succeeded")
        return result
    
    def update_response_status(
        self,
        dataset_name: str,
        record_id: str,
        new_status: str = "submitted",
        question_name: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace: Optional[str] = None
    ) -> bool:
        """
        更新记录的 Response 状态（draft -> submitted 或其他状态）
        
        Args:
            dataset_name: 数据集名称
            record_id: 记录ID
            new_status: 新状态 ("draft", "submitted", "discarded")
            question_name: 问题名称 (可选，如果不指定则更新所有问题的响应)
            user_id: 用户ID (可选，如果不指定则更新当前用户的响应)
            workspace: 工作区名称 (可选)
            
        Returns:
            True if successful, False otherwise
            
        Example:
            # 将指定记录的所有 draft 响应改为 submitted
            manager.update_response_status("my_dataset", "record_123", "submitted")
            
            # 只更新特定问题的响应状态
            manager.update_response_status(
                "my_dataset", 
                "record_123", 
                "submitted",
                question_name="label"
            )
        """
        # 获取数据集
        dataset = self.client.datasets(name=dataset_name)
        if not dataset:
            print(f"✗ Dataset '{dataset_name}' not found")
            return False
        
        # 验证workspace
        if workspace and hasattr(dataset, 'workspace') and dataset.workspace != workspace:
            print(f"✗ Dataset '{dataset_name}' not in workspace '{workspace}'")
            return False
        
        # 验证状态值
        valid_statuses = ["draft", "submitted", "discarded"]
        if new_status not in valid_statuses:
            print(f"✗ Invalid status '{new_status}'. Must be one of: {valid_statuses}")
            return False
        
        try:
            # 查找目标记录
            target_record = None
            for record in dataset.records:
                if str(record.id) == str(record_id):
                    target_record = record
                    break
            
            if not target_record:
                print(f"✗ Record with ID '{record_id}' not found in dataset '{dataset_name}'")
                return False
            
            # 获取当前用户ID（如果未指定）
            if user_id is None:
                user_id = self.client.me.id
            
            # 更新响应状态
            if not target_record.responses:
                print(f"⚠ No responses found for record '{record_id}'")
                return False
            
            updated_count = 0
            for resp in target_record.responses:
                # 检查是否匹配用户和问题
                if str(resp.user_id) == str(user_id):
                    if question_name is None or resp.question_name == question_name:
                        resp.status = new_status
                        updated_count += 1
            
            if updated_count == 0:
                print(f"⚠ No matching responses found to update")
                return False
            
            # 保存更新到服务器
            target_record.update()
            
            print(f"✓ Updated {updated_count} response(s) to status '{new_status}' for record '{record_id}'")
            return True
            
        except Exception as e:
            print(f"✗ Failed to update response status: {e}")
            return False


class ArgillaConnectionHelper:
    """Helper class for connection testing and validation"""
    
    @staticmethod
    def test_connection(
        api_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test connection to Argilla server
        
        Args:
            api_url: Argilla API URL
            api_key: Argilla API Key
            
        Returns:
            Dictionary with connection test results
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
        
        try:
            client = rg.Argilla._get_default()
            me = client.me
            
            result = {
                "success": True,
                "api_url": os.environ["ARGILLA_API_URL"],
                "username": me.username,
                "role": me.role,
                "message": "✓ Connection successful"
            }
            
            print(result["message"])
            return result
            
        except Exception as e:
            result = {
                "success": False,
                "api_url": os.environ.get("ARGILLA_API_URL", "Not set"),
                "error": str(e),
                "message": f"✗ Connection failed: {e}"
            }
            
            print(result["message"])
            return result
    
    @staticmethod
    def get_server_info() -> Dict[str, Any]:
        """
        Get Argilla server information
        
        Returns:
            Dictionary with server information
        """
        try:
            client = rg.Argilla._get_default()
            
            # Count resources
            dataset_count = sum(1 for _ in client.datasets)
            workspace_count = sum(1 for _ in client.workspaces)
            
            try:
                user_count = sum(1 for _ in client.users)
            except:
                user_count = "N/A (insufficient permissions)"
            
            info = {
                "api_url": os.environ.get("ARGILLA_API_URL", "Not set"),
                "current_user": client.me.username,
                "user_role": client.me.role,
                "dataset_count": dataset_count,
                "workspace_count": workspace_count,
                "user_count": user_count,
            }
            
            return info
            
        except Exception as e:
            print(f"✗ Failed to get server info: {e}")
            return {}


# Convenience functions
def create_workspace(name: str) -> Optional[rg.Workspace]:
    """Quick function to create a workspace"""
    manager = ArgillaWorkspaceManager()
    return manager.create_workspace(name)


def list_workspaces() -> List[Dict[str, Any]]:
    """Quick function to list all workspaces"""
    manager = ArgillaWorkspaceManager()
    return manager.list_workspaces()


def detect_annotation_conflicts(
    dataset_name: str,
    question_name: str,
    min_responses: int = 2,
    workspace: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Quick function to detect conflicts
    
    Args:
        dataset_name: Dataset name
        question_name: Question name
        min_responses: Minimum number of responses to check
        workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
    """
    manager = ArgillaRecordManager()
    return manager.detect_conflicts(dataset_name, question_name, min_responses, workspace)


def get_dataset_stats(dataset_name: str, question_name: str, workspace: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to get annotation statistics
    
    Args:
        dataset_name: Dataset name
        question_name: Question name
        workspace: Workspace name (optional, 避免误操作其他workspace的同名数据集)
    """
    manager = ArgillaRecordManager()
    return manager.get_annotation_stats(dataset_name, question_name, workspace)


def update_record(
    dataset_name: str,
    record_id: str,
    fields: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    suggestions: Optional[Dict[str, Any]] = None,
    workspace: Optional[str] = None
) -> bool:
    """
    Quick function to update a record
    
    Args:
        dataset_name: 数据集名称
        record_id: 记录ID
        fields: 要更新的字段内容
        metadata: 要更新的元数据
        suggestions: 要更新的建议标注
        workspace: 工作区名称 (optional, 避免误操作其他workspace的同名数据集)
    """
    manager = ArgillaRecordManager()
    return manager.update_record(dataset_name, record_id, fields, metadata, suggestions, workspace)


def submit_response(
    dataset_name: str,
    record_id: str,
    responses: Dict[str, Any],
    user_id: Optional[str] = None,
    workspace: Optional[str] = None
) -> bool:
    """
    Quick function to submit a response with submitted status
    
    Args:
        dataset_name: 数据集名称
        record_id: 记录ID
        responses: Response内容 {question_name: answer_value}
        user_id: 用户ID (可选)
        workspace: 工作区名称 (optional)
    """
    manager = ArgillaRecordManager()
    return manager.submit_response(dataset_name, record_id, responses, user_id, workspace)


def batch_submit_responses(
    dataset_name: str,
    records_data: List[Dict[str, Any]],
    workspace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to batch submit responses
    
    Args:
        dataset_name: 数据集名称
        records_data: 记录数据列表
        workspace: 工作区名称 (optional)
    """
    manager = ArgillaRecordManager()
    return manager.batch_submit_responses(dataset_name, records_data, workspace)


def update_response_status(
    dataset_name: str,
    record_id: str,
    new_status: str = "submitted",
    question_name: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace: Optional[str] = None
) -> bool:
    """
    Quick function to update response status (draft -> submitted)
    
    Args:
        dataset_name: 数据集名称
        record_id: 记录ID
        new_status: 新状态 ("draft", "submitted", "discarded")
        question_name: 问题名称 (可选)
        user_id: 用户ID (可选)
        workspace: 工作区名称 (optional)
    """
    manager = ArgillaRecordManager()
    return manager.update_response_status(dataset_name, record_id, new_status, question_name, user_id, workspace)


def get_annotator_progress(dataset_name: str, workspace: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to get annotator progress
    
    Args:
        dataset_name: 数据集名称
        workspace: 工作区名称 (optional)
    """
    manager = ArgillaRecordManager()
    return manager.get_annotator_progress(dataset_name, workspace)


def get_user_annotation_detail(
    dataset_name: str,
    user_id: Optional[str] = None,
    workspace: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Quick function to get user annotation details
    
    Args:
        dataset_name: 数据集名称
        user_id: 用户ID (可选，默认当前用户)
        workspace: 工作区名称 (optional)
    """
    manager = ArgillaRecordManager()
    return manager.get_user_annotation_detail(dataset_name, user_id, workspace)


def compare_annotators(
    dataset_name: str,
    question_name: str,
    workspace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to compare annotators
    
    Args:
        dataset_name: 数据集名称
        question_name: 问题名称
        workspace: 工作区名称 (optional)
    """
    manager = ArgillaRecordManager()
    return manager.compare_annotators(dataset_name, question_name, workspace)


def test_argilla_connection() -> bool:
    """Quick function to test connection"""
    result = ArgillaConnectionHelper.test_connection()
    return result["success"]


def print_server_info():
    """Quick function to print server information"""
    info = ArgillaConnectionHelper.get_server_info()
    
    if info:
        print("\n=== Argilla Server Information ===")
        print(f"API URL: {info['api_url']}")
        print(f"Current User: {info['current_user']} ({info['user_role']})")
        print(f"Datasets: {info['dataset_count']}")
        print(f"Workspaces: {info['workspace_count']}")
        print(f"Users: {info['user_count']}")
        print("=" * 35)


# Example usage
if __name__ == "__main__":
    # Test connection
    print("Testing Argilla connection...")
    test_argilla_connection()
    
    # Print server info
    print_server_info()
    
    # List workspaces
    print("\nWorkspaces:")
    workspaces = list_workspaces()
    for ws in workspaces:
        print(f"  - {ws['name']}")
    
    # Example: Detect conflicts in a dataset
    # conflicts = detect_annotation_conflicts("my_dataset", "label", min_responses=2)
    # print(f"\nFound {len(conflicts)} conflicts")
    
    # Example: Get annotation statistics
    # stats = get_dataset_stats("my_dataset", "label")
    # print(f"\nAnnotation Stats: {stats}")
    
    # Example: Update a record
    # update_record(
    #     "my_dataset",
    #     "record_id_123",
    #     fields={"text": "更新后的文本"},
    #     metadata={"category": "updated", "confidence": 0.95},
    #     suggestions={"label": "positive"}
    # )
    
    # Example: Submit a response with submitted status
    # submit_response(
    #     "my_dataset",
    #     "record_id_123",
    #     {"label": "positive"}
    # )
    
    # Example: Update response status from draft to submitted
    # update_response_status(
    #     "my_dataset",
    #     "record_id_123",
    #     "submitted"
    # )
    
    # Example: Batch submit responses
    # records = [
    #     {"record_id": "rec_1", "responses": {"label": "positive"}},
    #     {"record_id": "rec_2", "responses": {"label": "negative"}}
    # ]
    # result = batch_submit_responses("my_dataset", records)

