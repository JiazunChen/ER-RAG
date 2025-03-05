# Guide to Making Your First Submission

This document is designed to assist you in making your initial submission smoothly. Below, you'll find step-by-step instructions on specifying your software runtime and dependencies, structuring your code, and finally, submitting your project. Follow these guidelines to ensure a smooth submission process.

# Table of Contents

1. [Specifying Software Runtime and Dependencies](#specifying-software-runtime-and-dependencies)
2. [Code Structure Guidelines](#code-structure-guidelines)
3. [Submitting to Different Tracks](#submitting-to-different-tracks)
4. [Submission Entry Point](#submission-entry-point)
5. [Setting Up SSH Keys](#setting-up-ssh-keys)
6. [Managing Large Model Files with Git LFS](#managing-large-model-files-with-git-lfs)
    - [Why Use Git LFS?](#why-use-git-lfs)
    - [Steps to Use Git LFS](#steps-to-use-git-lfs)
    - [Handling Previously Committed Large Files](#handling-previously-committed-large-files)
7. [How to Submit Your Code](#how-to-submit-your-code)


## Specifying Software Runtime and Dependencies

Our platform supports custom runtime environments. This means you have the flexibility to choose any libraries or frameworks necessary for your project. Here’s how you can specify your runtime and dependencies:

- **`requirements.txt`**: List any PyPI packages your project needs.
- **`apt.txt`**: Include any apt packages required.
- **`Dockerfile`**: Optionally, you can provide your own Dockerfile. An example is located at `utilities/_Dockerfile`, which can serve as a helpful starting point.

For detailed setup instructions regarding runtime dependencies, refer to the documentation in the `docs/runtime.md` file.

## Code Structure Guidelines

Your project should follow the structure outlined in the starter kit. Here’s a brief overview of what each component represents:

```
.
├── README.md                       # Project documentation and setup instructions
├── .dockerignore                   # dockerignore file to exclude large models from image building phase
├── aicrowd.json                    # Submission meta information - like your username, track name
├── data
│   └── development.json            # Development dataset local testing
├── docs
│   └── runtime.md                  # Documentation on the runtime environment setup, dependency confifgs
├── local_evaluation.py             # Use this to check your model evaluation flow locally
├── models
│   ├── README.md                   # Documentation specific to the implementation of model interfaces
│   ├── base_model.py               # Base model class 
│   ├── dummy_model.py              # A simple or placeholder model for demonstration or testing
│   └── user_config.py              # IMPORTANT: Configuration file to specify your model 
├── requirements.txt                # Python packages to be installed for model development
└── Dockerfile                 # Example Dockerfile for specifying runtime via Docker
```

Remember, **your submission metadata JSON (`aicrowd.json`)** is crucial for mapping your submission to the challenge. Ensure it contains the correct `challenge_id`, `authors`, and other necessary information. To utilize GPUs, set the `"gpu": true` flag in your [aicrowd.json](../aicrowd.json) and `"gpu_count"` to a number between `1` and `4`.
For example, if you require 2 GPUS, you should set: 
```
    "gpu": true,
    "gpu_count": 2
```

## Submitting to Different Tracks

Specify the track by setting the appropriate `challenge_id` in your [aicrowd.json](aicrowd.json). Here are the challenge IDs for various tracks:

| Track Name                        | Challenge ID                                        |
|-----------------------------------|-----------------------------------------------------|
| Retrieval Summarization   | `meta-kdd-cup-24-crag-retrieval-summarization` |
| Knowledge Graph and Web Retrieval      | `meta-kdd-cup-24-crag-knowledge-graph-and-web-retrieval`    |
| End-to-end Retrieval Augmented Generation           | `meta-kdd-cup-24-crag-end-to-end-retrieval-augmented-generation`         |

## Submission Entry Point

The evaluation process will instantiate a model from [models/user_config.py](../models/user_config.py) for evaluation. Ensure this configuration is set correctly.

## Setting Up SSH Keys

You will have to add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).


## Managing Large Model Files with Git LFS

When preparing your submission, it's crucial to ensure all necessary models and files required by your inference code are properly saved and included. Due to the potentially large size of model weight files, we highly recommend using Git Large File Storage (Git LFS) to manage these files efficiently.

### Why Use Git LFS?

Git LFS is designed to handle large files more effectively than Git's default handling of large files. This ensures smoother operations and avoids common errors associated with large files, such as:

- `fatal: the remote end hung up unexpectedly`
- `remote: fatal: pack exceeds maximum allowed size`

These errors typically occur when large files are directly checked into the Git repository without Git LFS, leading to challenges in handling and transferring those files.

### Steps to Use Git LFS

1. **Install Git LFS**: If you haven't already, install Git LFS on your machine. Detailed instructions can be found [here](https://git-lfs.github.com/).

2. **Track Large Files**: Use Git LFS to track the large files within your project. You can do this by running `git lfs track "*.model"` (replace `*.model` with your file type).

3. **Add and Commit**: After tracking the large files with Git LFS, add and commit them as you would with any other file. Git LFS will automatically handle these files differently to optimize their storage and transfer.

4. **Push to Repository**: When you push your changes to the repository, Git LFS will manage the large files, ensuring a smooth push process.

### Handling Previously Committed Large Files

If you have already committed large files directly to your Git repository without using Git LFS, you may encounter issues. These files, even if not present in the current working directory, could still be in the Git history, leading to errors.

To resolve this, ensure that the large files are removed from the Git history and then re-add and commit them using Git LFS. This process cleans up the repository's history and avoids the aforementioned errors.

For more information on how to upload large files to your submission and detailed guidance on using Git LFS, please refer to [this detailed guide](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304).

**Note**: Properly managing large files not only facilitates smoother operations for you but also ensures that the evaluation process can proceed without hindrances.

## How to Submit Your Code

To submit your code, push a tag beginning with "submission-" to your repository on [GitLab](https://gitlab.aicrowd.com/). Follow these steps to make a submission:

1. Commit your changes with `git commit -am "Your commit message"`.
2. Tag your submission (e.g., `git tag -am "submission-v0.1" submission-v0.1`).
3. Push your changes and tags to the AIcrowd repository (replace `<YOUR_AICROWD_USER_NAME>` with your actual username).

After pushing your tag, you can view your submission details at `https://gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/meta-comphrehensive-rag-benchmark-starter-kit/issues`.

Ensure your `aicrowd.json` is correctly filled with the necessary metadata, and you've replaced `<YOUR_AICROWD_USER_NAME>` with your GitLab username in the provided URL.
