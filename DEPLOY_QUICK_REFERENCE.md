# 🚀 Quick Deployment Reference

## One-Page Quick Guide for Streamlit Cloud Deployment

### ⚡ 5-Minute Deployment

1. **Go to:** https://streamlit.io/cloud
2. **Sign in** with GitHub
3. **Click:** "New App"
4. **Fill in:**
   - Repository: `shivam2003-dev/ml-classification-comparison`
   - Branch: `main`
   - Main file: `app.py`
5. **Click:** "Deploy"
6. **Wait:** 2-5 minutes
7. **Done!** Your app is live 🎉

### 📋 Pre-Deployment Checklist

- [x] Code pushed to GitHub ✅
- [ ] Models trained (optional - app handles missing models)
- [ ] Test locally: `streamlit run app.py`
- [ ] All dependencies in `requirements.txt`

### 🔗 Your Links

- **GitHub:** https://github.com/shivam2003-dev/ml-classification-comparison
- **Streamlit Cloud:** https://streamlit.io/cloud
- **App URL:** `https://ml-classification-comparison.streamlit.app` (after deployment)

### 🔄 Update Your App

Just push to GitHub:
```bash
git add .
git commit -m "Update app"
git push origin main
```
Streamlit Cloud auto-deploys in 2-5 minutes!

### 🐛 Quick Troubleshooting

**App won't deploy?**
- Check `requirements.txt` has all packages
- Verify `app.py` exists
- Check deployment logs in Streamlit Cloud

**Models not loading?**
- Train models: `python train_models.py`
- Commit models: `git add model/ && git commit -m "Add models" && git push`
- Or app will show warning (handled gracefully)

**Need help?**
- See: `STREAMLIT_DEPLOYMENT.md` for detailed guide
- See: `DEPLOYMENT_CHECKLIST.md` for full checklist

---

**📖 Full Guides:**
- Detailed: `STREAMLIT_DEPLOYMENT.md`
- Checklist: `DEPLOYMENT_CHECKLIST.md`

