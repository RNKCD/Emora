from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="models/emora_inference",
    repo_id="r-onna-12/emora-emotion-classifier",
    repo_type="model",
)

print("Model uploaded successfully!")