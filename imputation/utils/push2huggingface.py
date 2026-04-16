from huggingface_hub import HfApi
import os


def _push_to_huggingface(local_path, repo_id, setting_name):

    api = HfApi()

    try:
        print(f"Uploading {local_path} to Hugging Face: {repo_id}...")

        api.create_repo(
            repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True
        )

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{setting_name}/cached_states.npy",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Auto-upload embedding: {setting_name}",
        )

        print(f"Upload successful: https://huggingface.co/datasets/{repo_id}")

        # if os.path.exists(local_path):
        #     os.remove(local_path)
        #     print(f"Deleted local file: {local_path}")

        parent_dir = os.path.dirname(local_path)
        # if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
        #     os.rmdir(parent_dir)
        #     print(f"Cleaned up empty directory: {parent_dir}")

    except Exception as e:
        print(f"HF Upload Failed: {str(e)}")
