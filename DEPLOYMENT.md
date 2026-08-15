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

## 3. Optional Vercel website

Do not upload the current Colab notebook as a Vercel Function. The notebook
mounts Google Drive and starts a temporary tunnel, while a hosted deployment
needs a normal web process and repo-relative files.

For the reliable split architecture:

    Browser -> Vercel landing page -> embedded Hugging Face Gradio Space
                                  -> DA-SwinSR inference

After the Hugging Face Space works, create a small static or Next.js site and
embed the permanent Space. Connect that frontend repository to Vercel and use
**Import Project**. The trained PyTorch inference remains on the Space; Vercel
serves branding, project explanation and the embedded demo.

Direct PyTorch inference inside a Vercel Python Function is possible only after
substantial restructuring and may require Vercel Large Functions because the
PyTorch dependency is much larger than the 6.88 MiB checkpoint. It also adds
CPU cold-start complexity, so it is not recommended for the hackathon demo.
