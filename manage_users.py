import os
from dotenv import load_dotenv
import argilla as rg
from argilla._exceptions import ArgillaError

# 1. 初始化连接 (Owner 权限)
load_dotenv()

if "ARGILLA_API_URL" not in os.environ:
    os.environ["ARGILLA_API_URL"] = "http://localhost:6900"
# 必须使用 Owner 的 API Key 才能管理用户和工作区
if "ARGILLA_API_KEY" not in os.environ:
    os.environ["ARGILLA_API_KEY"] = "owner.apikey"

print("正在连接到 Argilla...")
client = rg.Argilla._get_default()
print(f"当前登录用户: {client.me.username} (Role: {client.me.role})")

if client.me.role != "owner":
    print("警告: 当前 API Key 对应的用户不是 Owner，可能无法执行管理操作。")

# 2. 创建 Workspace
workspace_name = "labeling_team"
try:
    # 检查工作区是否存在
    workspace = client.workspaces(workspace_name)
    if workspace:
        print(f"Workspace '{workspace_name}' 已存在。")
    else:
        print(f"创建新 Workspace: {workspace_name}")
        workspace = rg.Workspace(name=workspace_name)
        workspace.create()
        print("Workspace 创建成功。")
except Exception as e:
    print(f"检查/创建 Workspace 时出错: {e}")
    exit(1)

# 3. 创建 User
username = "annotator_1"
password = "password1234"
first_name = "Annotator"
last_name = "One"

try:
    # 检查用户是否存在
    # Argilla 客户端的 users 属性通常是 Users 集合，支持按用户名索引或过滤
    # client.users(username) 返回 User 对象或 None (取决于版本)
    user = client.users(username)
    
    if user:
        print(f"用户 '{username}' 已存在。")
    else:
        print(f"创建新用户: {username}")
        # Role 可以是 'annotator', 'admin', 'owner'
        user = rg.User(
            username=username, 
            password=password, 
            role="annotator",
            first_name=first_name,
            last_name=last_name
        )
        user.create()
        print("用户创建成功。")

except Exception as e:
    print(f"检查/创建用户时出错: {e}")
    exit(1)

# 4. 将用户添加到 Workspace
# 在 Argilla 2.x 中，用户和 Workspace 的关系通过 workspace.add_user(user) 管理
try:
    # 重新获取 workspace 对象（确保是最新的）
    workspace = client.workspaces(workspace_name)
    user = client.users(username)
    
    # 检查用户是否已经在该 Workspace 中
    # 注意：users 属性可能是 list 或者需要再次 fetch
    # Argilla API 并没有直接列出 workspace 下所有 user 的简单属性，通常反向检查 user.workspaces
    # 或者直接尝试添加，如果已存在会报错或忽略
    
    print(f"正在将用户 '{username}' 添加到 Workspace '{workspace_name}'...")
    
    try:
        workspace.add_user(user.id)
        print(f"成功将用户 '{username}' 添加到 '{workspace_name}'。")
    except ArgillaError as e:
        if "already" in str(e).lower():
             print(f"用户 '{username}' 已经在 Workspace '{workspace_name}' 中。")
        else:
            # 某些版本的 API 可能没有直接抛错，而是静默成功
            # 或者通过 user.workspaces 检查
            print(f"添加用户操作完成 (API反馈: {e})")
    except Exception as e:
        # 兼容性处理：如果是旧版本 SDK，可能用法不同
        print(f"添加用户时发生异常: {e}")

    # 验证
    # 刷新 user 信息
    # user = client.users(username) # 某些属性可能不会立即刷新
    # 打印提示
    print("\n操作完成！")
    print("请尝试使用以下凭据登录 Argilla UI:")
    print(f"  URL: {os.environ['ARGILLA_API_URL']}")
    print(f"  Username: {username}")
    print(f"  Password: {password}")
    print(f"登录后，应该能看到 '{workspace_name}' 工作区。")
    print(f"注意: 如果数据集不在该工作区，Annotator 可能无法看到。")

except Exception as e:
    print(f"关联用户和 Workspace 时出错: {e}")

