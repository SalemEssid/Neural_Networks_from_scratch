# GitHub Push Guide

## Step-by-Step Instructions

### Option 1: Create New GitHub Repository (Recommended)

#### 1. Create Repository on GitHub
- Go to https://github.com/new
- Repository name: `neural-network-framework` (or your preferred name)
- Description: `Deep Neural Network Framework from Scratch with MNIST Classification`
- Choose: Public (to share) or Private
- DO NOT initialize with README (we have one)
- Click "Create repository"

#### 2. Get Your Repository URL
Copy the URL shown: `https://github.com/[YOUR-USERNAME]/neural-network-framework.git`

#### 3. Configure Git Locally

Run these commands in your project directory:

```bash
# Navigate to project
cd c:\Users\salem\.vscode\PY\NNs

# Check git status
git status

# Configure git (if first time)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Deep Neural Network Framework with MNIST classification

- Implements neural network from scratch using NumPy
- 4 optimizers: SGD, Momentum, RMSprop, Adam
- Regularization: L1, L2, Elastic Net, Dropout
- Flexible architecture configuration
- Automatic experiment tracking and visualization
- Comprehensive evaluation metrics"

# Add remote repository
git remote add origin https://github.com/SalemEssid/Neural_Networks_from_scratch.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Option 2: If Repository Already Exists

```bash
# Add remote
git remote add origin https://github.com/SalemEssid/Neural_Networks_from_scratch.git

# Push existing commits
git push -u origin main
```

---

## What Gets Pushed?

### ✅ Included (Important Files)
```
src/                    # All source code
configs/               # Configuration files
main.py               # Entry point
README.md             # Documentation
.gitignore            # Git ignore rules
```

### ❌ Excluded (Via .gitignore)
```
.venv/                # Virtual environment
__pycache__/          # Python cache
experiments/pkl files # Large model files (keep folder structure)
*.log                 # Log files
.vscode/             # IDE settings
```

---

## After Pushing

### View on GitHub
- Go to: `https://github.com/[YOUR-USERNAME]/neural-network-framework`
- Click "Code" to see your files
- Click "Commits" to see your history

### Share Your Project
- Use the repository URL in README, resume, portfolio
- Add topics: `neural-networks`, `machine-learning`, `deep-learning`, `mnist`, `python`

---

## Commit Message Template

For future commits, use this format:

```
git commit -m "Brief summary (50 chars max)

- Detailed change 1
- Detailed change 2
- Detailed change 3"
```

---

## Essential Commands Reference

| Command | Purpose |
|---------|---------|
| `git status` | Check changed files |
| `git add .` | Stage all changes |
| `git commit -m "msg"` | Commit with message |
| `git push` | Push to GitHub |
| `git pull` | Get latest from GitHub |
| `git log` | View commit history |

---

## Troubleshooting

### Error: "fatal: not a git repository"
```bash
git init
```

### Error: "Authentication failed"
- Use personal access token instead of password
- Or configure SSH keys

### Error: "Please tell me who you are"
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Next Steps

1. ✅ Push code to GitHub
2. Add topics/tags to repository
3. Share with others
4. Enable GitHub Pages (optional)
5. Add GitHub Actions CI/CD (optional)

---

**Ready to push? Replace [YOUR-USERNAME] in the commands above and run them!**
