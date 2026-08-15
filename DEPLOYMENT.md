# Deployment instructions

## 1. Push safely to GitHub

Keep the repository private until the hackathon rules and dataset/model
redistribution permissions have been confirmed. This folder intentionally does
not contain the training dataset, ground truth, competition inputs or the two
large resume checkpoints.

Create an empty private repository named `da-swinsr-demo` on GitHub. Do not add
a README, license or `.gitignore` on the GitHub creation page because those
files already exist locally.

Open PowerShell in this exact folder and run:

    git init -b main
    git add .
    git status
    git commit -m "Add DA-SwinSR inference demo"
    git remote add origin https://github.com/YOUR_USERNAME/da-swinsr-demo.git
    git push -u origin main

Inspect `git status` before committing. It must not show `train`, `NoisyLR`,
`GT`, ZIP archives, tokens or generated predictions.

Never run these commands from the parent `Dataset_semicon` folder.

## 2. Recommended permanent model hosting: Hugging Face Spaces

1. Create a Hugging Face account and choose **New Space**.
2. Name it `da-swinsr-demo` and select **Gradio** as the SDK.
3. Choose public/private visibility according to the hackathon rules.
4. Upload the contents of this repository, preserving the `weights/` and
   `metadata/` directories, or push them to the Space's Git repository.
5. Wait for the Space to build. It will start `app.py` automatically.
6. Test it with one valid 128 x 128 `.npy` file from `NoisyLR`.

Free-host availability and hardware eligibility depend on the current Hugging
Face account policy. CPU inference is supported by this app; GPU is optional.

## 3. Vercel deployment

Do not upload the current Colab notebook as a Vercel Function. The notebook
mounts Google Drive and starts a temporary tunnel, while a hosted deployment
needs a normal web process and repo-relative files.

This repository also supports direct CPU inference on Vercel through the root
`index.py` FastAPI entrypoint. The browser sends one `.npy` payload to
`/api/restore` and receives all previews and downloads in the same response, so
no temporary server files or stateful queue are required. PyTorch expands the
uncompressed Python Function bundle to about 922 MB, although the checkpoint
itself is only 6.88 MiB.

In the Vercel project:

1. Keep **Fluid Compute** enabled.
2. Add the Production and Preview environment variable
   `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`.
3. Deploy the repository with Python 3.12.
4. Verify that `/` returns HTTP 200, then run one valid 128 x 128 `.npy` input
   and confirm that all three previews and both downloads appear.

Do not add a catch-all rewrite to `/api/index`. Current Vercel FastAPI
detection serves the root `index.py` application directly; that legacy rewrite
causes every page to return 404.

For a split architecture with smaller Vercel cold starts:

    Browser -> Vercel landing page -> embedded Hugging Face Gradio Space
                                  -> DA-SwinSR inference

After the Hugging Face Space works, create a small static or Next.js site and
embed the permanent Space. Connect that frontend repository to Vercel and use
**Import Project**. The trained PyTorch inference remains on the Space; Vercel
serves branding, project explanation and the embedded demo.

The direct Vercel deployment uses Large Functions and can have a noticeable
CPU cold start. Hugging Face Spaces remains a useful alternative if the team
prefers first-class Gradio hosting.
