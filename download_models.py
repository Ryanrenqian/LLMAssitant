from huggingface_hub import snapshot_download
snapshot_download(resume_download=True,repo_id='tiiuae/falcon-40b-instruct',local_dir_use_symlinks=False,local_dir='/root/autodl-tmp/cache/transformers')