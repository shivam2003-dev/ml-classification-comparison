# 🚀 Streamlit Cloud Deployment Guide

Complete step-by-step guide to deploy your ML Classification Models app on Streamlit Community Cloud (FREE).

## 📋 Prerequisites

- ✅ GitHub repository created and code pushed
- ✅ Repository is public (for free tier) or you have Streamlit Cloud Pro
- ✅ All code files are in the repository
- ✅ `requirements.txt` exists with all dependencies
- ✅ Main app file (`app.py` or `streamlit_app.py`) exists

## 🎯 Step-by-Step Deployment

### Step 1: Access Streamlit Cloud

1. Go to **https://streamlit.io/cloud**
2. Click **"Sign in"** or **"Get started"**
3. Sign in with your **GitHub account**
   - This will authorize Streamlit Cloud to access your repositories

### Step 2: Create New App

1. After signing in, you'll see the Streamlit Cloud dashboard
2. Click the **"New app"** button (usually in the top right or center)
3. You'll see a form to configure your app

### Step 3: Configure App Settings

Fill in the deployment form:

#### Repository Selection
- **Repository**: Select `shivam2003-dev/ml-classification-comparison`
  - If you don't see it, make sure:
    - Repository is public (for free tier)
    - You've authorized Streamlit Cloud to access your GitHub account
    - Repository exists and has been pushed

#### Branch Selection
- **Branch**: Select `main` (or `master` if that's your default branch)
  - This is the branch Streamlit Cloud will deploy from

#### Main File Path
- **Main file path**: Enter `app.py`
  - Alternative: `streamlit_app.py` (both work, `app.py` is preferred for Streamlit Cloud)
  - This is the entry point of your Streamlit application

#### Advanced Settings (Optional)
- **Python version**: Leave default (usually 3.10 or 3.11)
- **App URL**: Leave default (will be auto-generated)
  - Format: `https://your-app-name.streamlit.app`

### Step 4: Deploy

1. Review all settings
2. Click **"Deploy"** button
3. Wait for deployment (usually 2-5 minutes)

### Step 5: Monitor Deployment

You'll see a deployment log showing:
- ✅ Installing dependencies from `requirements.txt`
- ✅ Building the app
- ✅ Starting the app
- ✅ App is live!

**Common Status Messages:**
- 🟡 **"Deploying..."** - App is being built
- 🟢 **"Running"** - App is live and accessible
- 🔴 **"Error"** - Check the logs for issues

### Step 6: Access Your App

Once deployed, you'll see:
- **App URL**: `https://ml-classification-comparison.streamlit.app` (or similar)
- Click the URL or the **"Open app"** button to view your live app

## 🔧 Troubleshooting

### Issue: "App failed to deploy"

**Common Causes & Solutions:**

1. **Missing dependencies in requirements.txt**
   - ✅ Check that all packages are listed
   - ✅ Verify version numbers are correct
   - ✅ Check the deployment logs for specific missing packages

2. **Import errors**
   - ✅ Ensure all imports are available in requirements.txt
   - ✅ Check for typos in import statements
   - ✅ Verify file paths are correct

3. **Model files not found**
   - ✅ Models need to be committed to GitHub (or handle missing models gracefully)
   - ✅ Check that `model/` directory structure is correct
   - ✅ Verify file paths in your code

4. **Python version mismatch**
   - ✅ Check Python version in requirements (if specified)
   - ✅ Ensure code is compatible with Python 3.8+

### Issue: "App loads but models don't work"

**Solutions:**
1. **Models not in repository**
   - Train models locally: `python train_models.py`
   - Commit models: `git add model/ && git commit -m "Add models" && git push`
   - Redeploy on Streamlit Cloud

2. **Model files too large**
   - Streamlit Cloud free tier has size limits
   - Consider using smaller models or model compression
   - Alternative: Load models from external storage (S3, etc.)

3. **Path issues**
   - Use relative paths: `model/logistic_regression.pkl`
   - Don't use absolute paths
   - Check that paths match your repository structure

### Issue: "App is slow to load"

**Solutions:**
1. Use `@st.cache_data` for model loading (already implemented)
2. Optimize model sizes
3. Consider lazy loading of models

## 📝 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] All code is pushed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] `app.py` or `streamlit_app.py` exists and works locally
- [ ] Models are trained (or app handles missing models gracefully)
- [ ] No hardcoded paths (use relative paths)
- [ ] No API keys or secrets in code (use Streamlit secrets if needed)
- [ ] Repository is public (for free tier)
- [ ] Test app locally: `streamlit run app.py`

## 🔄 Updating Your App

### Automatic Updates
- Streamlit Cloud automatically redeploys when you push to the main branch
- Just push your changes: `git push origin main`
- Wait 2-5 minutes for redeployment

### Manual Redeploy
1. Go to Streamlit Cloud dashboard
2. Find your app
3. Click **"⋮"** (three dots) menu
4. Select **"Redeploy"**

## 🎨 Custom Domain (Optional)

For free tier, you get:
- URL format: `https://your-app-name.streamlit.app`

For custom domain, you need Streamlit Cloud Pro.

## 📊 Monitoring & Logs

### View Logs
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click **"Manage app"**
4. View **"Logs"** tab for:
   - Deployment logs
   - Runtime logs
   - Error messages

### App Metrics
- View app usage statistics
- Monitor performance
- Check for errors

## 🔐 Secrets Management

If you need to store API keys or secrets:

1. Go to Streamlit Cloud dashboard
2. Click **"Settings"** → **"Secrets"**
3. Add secrets in TOML format:
   ```toml
   [secrets]
   api_key = "your-api-key"
   ```
4. Access in code: `st.secrets["api_key"]`

## 📱 App Configuration

### Streamlit Config File (Optional)

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

## ✅ Post-Deployment Verification

After deployment, verify:

1. **App loads successfully**
   - ✅ No errors on initial load
   - ✅ All pages accessible

2. **Models work (if trained)**
   - ✅ Model comparison page shows metrics
   - ✅ Can select different models
   - ✅ Confusion matrices display correctly

3. **File upload works**
   - ✅ Can upload CSV files
   - ✅ Predictions are generated
   - ✅ Download predictions works

4. **UI elements work**
   - ✅ Navigation sidebar works
   - ✅ All buttons functional
   - ✅ Charts and visualizations display

## 🎯 Quick Deployment Commands

```bash
# 1. Ensure everything is committed
git status

# 2. Push to GitHub
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main

# 3. Go to https://streamlit.io/cloud and deploy
# 4. Wait for deployment to complete
# 5. Test your live app!
```

## 📞 Support Resources

- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Streamlit Forum**: https://discuss.streamlit.io
- **GitHub Issues**: Report bugs in your repository

## 🎉 Success!

Once deployed, you'll have:
- ✅ Live web application
- ✅ Shareable URL
- ✅ Automatic updates on git push
- ✅ Free hosting (Community Cloud)

**Your app URL will be something like:**
`https://ml-classification-comparison.streamlit.app`

Share this link in your assignment submission!

---

**Note:** For this assignment, make sure to:
1. Deploy the app before submission deadline
2. Test all features work correctly
3. Include the live app URL in your PDF submission
4. Keep the app accessible for evaluation period

