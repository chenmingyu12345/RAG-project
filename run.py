"""
Laike RAG 统一启动脚本

同时启动两个独立服务：
  - 用户端 :8000   (聊天问答 + Webhook)
  - 管理端 :8001   (后台管理)
"""
import os
import sys
import threading
from dotenv import load_dotenv

load_dotenv()


def start_user():
    """启动用户端服务"""
    import uvicorn
    import web_service

    port = int(os.getenv("WEB_PORT", 8000))
    host = os.getenv("WEB_HOST", "0.0.0.0")
    print(f"\n  [用户端] 启动中... http://{host}:{port}")
    uvicorn.run(web_service.app, host=host, port=port, log_level="info")


def start_admin():
    """启动管理端服务"""
    import uvicorn
    import admin_service

    port = int(os.getenv("ADMIN_PORT", 8001))
    host = os.getenv("WEB_HOST", "0.0.0.0")
    print(f"  [管理端] 启动中... http://{host}:{port}")
    uvicorn.run(admin_service.app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  Laike RAG - 双服务启动")
    print("=" * 56)

    user_port = os.getenv("WEB_PORT", "8000")
    admin_port = os.getenv("ADMIN_PORT", "8001")

    print(f"  用户端: http://localhost:{user_port}    (AI 问答)")
    print(f"  管理端: http://localhost:{admin_port}    (后台管理 + Webhook)")
    print(f"  按 Ctrl+C 停止所有服务\n")

    t_user = threading.Thread(target=start_user, daemon=True)
    t_admin = threading.Thread(target=start_admin, daemon=True)

    t_user.start()
    t_admin.start()

    try:
        t_user.join()
        t_admin.join()
    except KeyboardInterrupt:
        print("\n  服务已停止")
        sys.exit(0)
