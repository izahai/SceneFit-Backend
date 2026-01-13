# SceneFit-Backend
A scene-aware system that retrieves suitable clothing based on environmental context.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

## Run on Google Colab

Use the provided `run_colab.ipynb` notebook to start the backend in Colab.

1) Create a Drive shortcut for the dataset folder:
   `https://drive.google.com/drive/folders/1AAHqvWLGTxXsRxc85inFpjssEHwWWQsy?dmr=1&ec=wgc-drive-globalnav-goto`

2) Run the file `run_colab.ipynb` in Colab.

If you use the diffusion endpoint with SD3.5, set your Hugging Face token
in the Colab environment before starting the server:
`export HF_TOKEN=your_token_here`
