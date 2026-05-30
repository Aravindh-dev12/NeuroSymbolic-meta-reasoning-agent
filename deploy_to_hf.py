"""
deploy_to_hf.py — Programmatically deploy the NeuroSymbolic agent to Hugging Face Spaces.
Uses huggingface_hub to push code and assets, auto-configuring Gradio hosting under user account.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    print("====================================================")
    print("🧬 Hugging Face Spaces Programmatic Deployment Helper")
    print("====================================================\n")

    # Step 1: Authentication
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("💡 Hugging Face Write Token (HF_TOKEN) not found in environment.")
        hf_token = input("🔑 Please enter your Hugging Face WRITE token: ").strip()
        if not hf_token:
            print("❌ Error: A valid Hugging Face token is required for deployment.")
            sys.exit(1)

    try:
        login(token=hf_token, write_permission=True)
        print("✅ Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    api = HfApi()
    
    # Step 2: Get user info to determine namespace
    try:
        user_info = api.whoami(token=hf_token)
        username = user_info["name"]
        print(f"👤 Detected Hugging Face user: {username}")
    except Exception as e:
        print(f"❌ Failed to fetch user profile: {e}")
        sys.exit(1)

    # Step 3: Get Space Name
    default_space_name = "NeuroSymbolic-Meta-Reasoner"
    space_name = input(f"📁 Enter Space Name [{default_space_name}]: ").strip()
    if not space_name:
        space_name = default_space_name

    repo_id = f"{username}/{space_name}"
    print(f"🚀 Deploying to space: https://huggingface.co/spaces/{repo_id}")

    # Step 4: Create Space if it doesn't exist
    try:
        print(f"🛠️ Creating Hugging Face Space '{repo_id}' (SDK: gradio)...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
            token=hf_token
        )
        print("✅ Space repository created/verified.")
    except Exception as e:
        print(f"❌ Failed to create Space repository: {e}")
        sys.exit(1)

    # Step 5: Upload folder with exclusions
    print("📦 Preparing workspace files...")
    ignore_patterns = [
        "venv/**",
        ".git/**",
        ".github/**",
        "logs/**",
        "data/**",
        "models/**",
        "cache/**",
        "__pycache__/**",
        "**/*.pyc",
        "**/.DS_Store",
        ".env"
    ]
    
    try:
        print("📤 Uploading files to Hugging Face Spaces (this might take a few moments)...")
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            token=hf_token
        )
        print("\n🎉 SUCCESS! Deployment complete.")
        print(f"🌐 Access your live web dashboard here: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
