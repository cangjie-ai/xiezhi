"""
Argilla User Management Utilities

This module provides utility functions for managing Argilla users, including:
- Creating, deleting, and updating users
- Changing passwords and roles
- Managing user-workspace relationships
- Listing and querying users

Author: Argilla Utils
Version: 1.0.0
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import argilla as rg
from argilla._exceptions import ArgillaError
import httpx


class ArgillaUserManager:
    """Manager class for Argilla user operations"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Argilla User Manager
        
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
        
    def get_current_user(self) -> Dict[str, Any]:
        """
        Get current logged in user information
        
        Returns:
            Dictionary with user information (username, role, etc.)
        """
        me = self.client.me
        return {
            "username": me.username,
            "role": me.role,
            "id": str(me.id) if hasattr(me, 'id') else None,
            "first_name": me.first_name if hasattr(me, 'first_name') else None,
            "last_name": me.last_name if hasattr(me, 'last_name') else None,
        }
    
    def create_user(
        self, 
        username: str, 
        password: str,
        role: str = "annotator",
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> rg.User:
        """
        Create a new Argilla user
        
        Args:
            username: Unique username
            password: User password
            role: User role ('annotator', 'admin', or 'owner')
            first_name: User's first name (optional)
            last_name: User's last name (optional)
            
        Returns:
            Created User object
            
        Raises:
            ArgillaError: If user already exists or creation fails
        """
        if role not in ["annotator", "admin", "owner"]:
            raise ValueError(f"Invalid role: {role}. Must be 'annotator', 'admin', or 'owner'")
        
        # Check if user exists
        existing_user = self.get_user(username)
        if existing_user:
            raise ArgillaError(f"User '{username}' already exists")
        
        # Create user
        user = rg.User(
            username=username,
            password=password,
            role=role,
            first_name=first_name,
            last_name=last_name
        )
        user.create()
        
        print(f"✓ User '{username}' created successfully with role '{role}'")
        return user
    
    def delete_user(self, username: str) -> bool:
        """
        Delete an Argilla user
        
        Args:
            username: Username to delete
            
        Returns:
            True if successful, False otherwise
        """
        user = self.get_user(username)
        if not user:
            print(f"✗ User '{username}' not found")
            return False
        
        try:
            user.delete()
            print(f"✓ User '{username}' deleted successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to delete user '{username}': {e}")
            return False
    
    def get_user(self, username: str) -> Optional[rg.User]:
        """
        Get user by username
        
        Args:
            username: Username to search for
            
        Returns:
            User object if found, None otherwise
        """
        try:
            user = self.client.users(username)
            return user
        except Exception:
            return None
    
    def list_users(self) -> List[Dict[str, Any]]:
        """
        List all users in the system
        
        Returns:
            List of dictionaries containing user information
        """
        try:
            users = []
            for user in self.client.users:
                users.append({
                    "id": str(user.id),
                    "username": user.username,
                    "role": user.role,
                    "first_name": user.first_name if hasattr(user, 'first_name') else None,
                    "last_name": user.last_name if hasattr(user, 'last_name') else None,
                })
            return users
        except Exception as e:
            print(f"✗ Failed to list users: {e}")
            return []
    
    def update_user_password(self, username: str, new_password: str) -> bool:
        """
        Update user password
        
        Args:
            username: Username to update
            new_password: New password
            
        Returns:
            True if successful, False otherwise
            
        Note:
            This method uses the HTTP client directly because the user.update() 
            method doesn't support password updates (returns 405 error)
        """
        user = self.get_user(username)
        if not user:
            print(f"✗ User '{username}' not found")
            return False
        
        try:
            # Method 1: Try using Argilla's internal HTTP client
            if hasattr(self.client, 'http_client'):
                http_client = self.client.http_client
                response = http_client.patch(
                    f"/api/v1/users/{user.id}",
                    json={"password": new_password}
                )
                
                if response.status_code in [200, 204]:
                    print(f"✓ Password updated for user '{username}'")
                    return True
                else:
                    print(f"✗ Failed to update password: HTTP {response.status_code} - {response.text}")
                    return False
            
            # Method 2: Fallback to httpx direct call
            else:
                api_url = os.environ.get("ARGILLA_API_URL", "http://localhost:6900")
                api_key = os.environ.get("ARGILLA_API_KEY", "owner.apikey")
                
                with httpx.Client() as client:
                    response = client.patch(
                        f"{api_url}/api/v1/users/{user.id}",
                        json={"password": new_password},
                        headers={"X-Argilla-Api-Key": api_key}
                    )
                    
                    if response.status_code in [200, 204]:
                        print(f"✓ Password updated for user '{username}'")
                        return True
                    else:
                        print(f"✗ Failed to update password: HTTP {response.status_code} - {response.text}")
                        return False
                        
        except Exception as e:
            print(f"✗ Failed to update password for '{username}': {e}")
            return False
    
    def update_user_role(self, username: str, new_role: str) -> bool:
        """
        Update user role
        
        Args:
            username: Username to update
            new_role: New role ('annotator', 'admin', or 'owner')
            
        Returns:
            True if successful, False otherwise
        """
        if new_role not in ["annotator", "admin", "owner"]:
            raise ValueError(f"Invalid role: {new_role}. Must be 'annotator', 'admin', or 'owner'")
        
        user = self.get_user(username)
        if not user:
            print(f"✗ User '{username}' not found")
            return False
        
        try:
            old_role = user.role
            user.role = new_role
            user.update()
            print(f"✓ User '{username}' role updated from '{old_role}' to '{new_role}'")
            return True
        except Exception as e:
            print(f"✗ Failed to update role for '{username}': {e}")
            return False
    
    def update_user_info(
        self, 
        username: str, 
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> bool:
        """
        Update user information (first name, last name)
        
        Args:
            username: Username to update
            first_name: New first name (optional)
            last_name: New last name (optional)
            
        Returns:
            True if successful, False otherwise
        """
        user = self.get_user(username)
        if not user:
            print(f"✗ User '{username}' not found")
            return False
        
        try:
            if first_name is not None:
                user.first_name = first_name
            if last_name is not None:
                user.last_name = last_name
            
            user.update()
            print(f"✓ User '{username}' information updated")
            return True
        except Exception as e:
            print(f"✗ Failed to update info for '{username}': {e}")
            return False
    
    def add_user_to_workspace(self, username: str, workspace_name: str) -> bool:
        """
        Add user to a workspace
        
        Args:
            username: Username to add
            workspace_name: Workspace name
            
        Returns:
            True if successful, False otherwise
        """
        user = self.get_user(username)
        if not user:
            print(f"✗ User '{username}' not found")
            return False
        
        workspace = self.client.workspaces(workspace_name)
        if not workspace:
            print(f"✗ Workspace '{workspace_name}' not found")
            return False
        
        try:
            workspace.add_user(user.id)
            print(f"✓ User '{username}' added to workspace '{workspace_name}'")
            return True
        except ArgillaError as e:
            if "already" in str(e).lower():
                print(f"ℹ User '{username}' is already in workspace '{workspace_name}'")
                return True
            else:
                print(f"✗ Failed to add user to workspace: {e}")
                return False
        except Exception as e:
            print(f"✗ Failed to add user to workspace: {e}")
            return False
    
    def remove_user_from_workspace(self, username: str, workspace_name: str) -> bool:
        """
        Remove user from a workspace
        
        Args:
            username: Username to remove
            workspace_name: Workspace name
            
        Returns:
            True if successful, False otherwise
        """
        user = self.get_user(username)
        if not user:
            print(f"✗ User '{username}' not found")
            return False
        
        workspace = self.client.workspaces(workspace_name)
        if not workspace:
            print(f"✗ Workspace '{workspace_name}' not found")
            return False
        
        try:
            workspace.remove_user(user.id)
            print(f"✓ User '{username}' removed from workspace '{workspace_name}'")
            return True
        except Exception as e:
            print(f"✗ Failed to remove user from workspace: {e}")
            return False
    
    def get_user_workspaces(self, username: str) -> List[str]:
        """
        Get list of workspaces that a user belongs to
        
        Args:
            username: Username to query
            
        Returns:
            List of workspace names
        """
        user = self.get_user(username)
        if not user:
            print(f"✗ User '{username}' not found")
            return []
        
        try:
            # Note: This might vary based on Argilla version
            # Some versions expose user.workspaces, others require reverse lookup
            workspaces = []
            for ws in self.client.workspaces:
                # Check if user is in workspace
                try:
                    ws_users = list(ws.users)
                    if any(u.id == user.id for u in ws_users):
                        workspaces.append(ws.name)
                except:
                    # If can't list users, try alternate method
                    pass
            
            return workspaces
        except Exception as e:
            print(f"✗ Failed to get workspaces for '{username}': {e}")
            return []


# Convenience functions for quick operations
def create_user(username: str, password: str, role: str = "annotator", 
                first_name: Optional[str] = None, last_name: Optional[str] = None) -> Optional[rg.User]:
    """Quick function to create a user"""
    manager = ArgillaUserManager()
    try:
        return manager.create_user(username, password, role, first_name, last_name)
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def delete_user(username: str) -> bool:
    """Quick function to delete a user"""
    manager = ArgillaUserManager()
    return manager.delete_user(username)


def change_password(username: str, new_password: str) -> bool:
    """Quick function to change user password"""
    manager = ArgillaUserManager()
    return manager.update_user_password(username, new_password)


def change_role(username: str, new_role: str) -> bool:
    """Quick function to change user role"""
    manager = ArgillaUserManager()
    return manager.update_user_role(username, new_role)


def add_to_workspace(username: str, workspace_name: str) -> bool:
    """Quick function to add user to workspace"""
    manager = ArgillaUserManager()
    return manager.add_user_to_workspace(username, workspace_name)


def list_all_users() -> List[Dict[str, Any]]:
    """Quick function to list all users"""
    manager = ArgillaUserManager()
    return manager.list_users()


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ArgillaUserManager()
    
    # Show current user
    print("Current user:", manager.get_current_user())
    
    # List all users
    print("\nAll users:")
    users = manager.list_users()
    for user in users:
        print(f"  - {user['username']} ({user['role']})")
    
    # Example: Create a new user
    # manager.create_user("test_user", "password123", role="annotator", first_name="Test", last_name="User")
    
    # Example: Add user to workspace
    # manager.add_user_to_workspace("test_user", "my_workspace")
    
    # Example: Change user role
    # manager.update_user_role("test_user", "admin")
    
    # Example: Delete user
    # manager.delete_user("test_user")


