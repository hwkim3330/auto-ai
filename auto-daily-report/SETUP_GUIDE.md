# Setup Guide - Auto Daily Report

Complete setup instructions for deploying the Auto Daily Report system.

---

## Prerequisites

- Python 3.8 or higher
- Git
- GitHub account
- API keys (see below)

---

## 1. Local Setup

### Step 1: Clone and Setup Environment

```bash
# Navigate to project directory
cd auto-daily-report

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get API Keys

You need to obtain API keys from these services:

#### Required APIs:

1. **OpenAI API** (for GPT-4 summarization)
   - Go to: https://platform.openai.com/api-keys
   - Create account / Login
   - Click "Create new secret key"
   - Copy the key (starts with `sk-...`)
   - **Cost**: ~$0.01 per daily report

2. **NewsAPI** (for news collection)
   - Go to: https://newsapi.org/register
   - Register with email
   - Copy API key from dashboard
   - **Free tier**: 100 requests/day (enough for daily reports)

3. **OpenWeatherMap** (for weather data)
   - Go to: https://openweathermap.org/api
   - Create account
   - Go to API keys tab
   - Copy default API key
   - **Free tier**: 1,000 calls/day

#### Optional APIs:

4. **Anthropic Claude** (alternative to GPT-4)
   - Go to: https://console.anthropic.com/
   - Create account and get API key
   - Only needed if using Claude instead of GPT-4

5. **Telegram Bot** (for notifications)
   - Open Telegram app
   - Search for @BotFather
   - Send `/newbot` command
   - Follow instructions to create bot
   - Copy bot token
   - Get your chat ID:
     - Message your bot
     - Go to: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
     - Find your `chat.id` in the response

### Step 3: Configure Environment Variables

```bash
# Copy example file
cp .env.example .env

# Edit .env file
nano .env  # or use any text editor
```

**Required variables** in `.env`:
```env
OPENAI_API_KEY=sk-your_openai_key_here
NEWS_API_KEY=your_newsapi_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
```

**Optional variables**:
```env
ANTHROPIC_API_KEY=your_anthropic_key_here
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
AI_MODEL=gpt-4  # or gpt-3.5-turbo, claude-3-sonnet-20240229
```

### Step 4: Test Run

```bash
# Run the report generator
python main.py
```

If successful, you should see:
- News collected
- Weather data retrieved
- AI summaries generated
- Reports saved in `reports/daily/`

Check the generated files:
```bash
ls -lh reports/daily/
cat reports/daily/report_*.md
```

---

## 2. GitHub Setup

### Step 1: Create GitHub Repository

**Option A: Using GitHub CLI (recommended)**
```bash
# Install GitHub CLI first (https://cli.github.com/)
gh auth login

# Create and push repository
gh repo create auto-daily-report --public --source=. --push
```

**Option B: Manual Setup**
1. Go to https://github.com/new
2. Repository name: `auto-daily-report`
3. Description: "Automated daily news and weather reporting system"
4. Make it Public (for GitHub Actions free tier)
5. Don't initialize with README (we already have one)
6. Click "Create repository"

Then push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/auto-daily-report.git
git branch -M main
git push -u origin main
```

### Step 2: Configure GitHub Secrets

GitHub Actions needs API keys to run. Store them as secrets:

1. Go to your repository on GitHub
2. Click **Settings** tab
3. In left sidebar: **Secrets and variables** → **Actions**
4. Click **New repository secret**

Add these secrets:

| Name | Value | Required |
|------|-------|----------|
| `OPENAI_API_KEY` | Your OpenAI key | ✅ Yes |
| `NEWS_API_KEY` | Your NewsAPI key | ✅ Yes |
| `OPENWEATHER_API_KEY` | Your OpenWeather key | ✅ Yes |
| `ANTHROPIC_API_KEY` | Your Anthropic key | ❌ Optional |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token | ❌ Optional |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID | ❌ Optional |
| `AI_MODEL` | `gpt-4` or `gpt-3.5-turbo` | ❌ Optional |

### Step 3: Enable GitHub Actions

1. Go to **Actions** tab in your repository
2. Click "I understand my workflows, go ahead and enable them"
3. The workflow will run:
   - **Automatically**: Every day at 7:00 AM KST (22:00 UTC)
   - **Manually**: Click "Run workflow" button

### Step 4: Verify First Run

**Manual test run:**
1. Go to **Actions** tab
2. Click "Daily Report Generation" workflow
3. Click "Run workflow" dropdown
4. Click green "Run workflow" button
5. Wait 1-2 minutes
6. Check the run status and logs

**Expected output:**
- Green checkmark = Success
- Reports committed to repository
- Files in `reports/daily/` updated

---

## 3. Configuration

### Customize News Sources

Edit `config/config.yaml`:

```yaml
news:
  # Add/remove news sources
  sources:
    - bbc-news
    - techcrunch
    - the-verge

  # Add keywords you care about
  keywords:
    - AI
    - 우주
    - 기후변화

  # Exclude topics
  exclude:
    - 연예
    - 스포츠
```

### Change Report Schedule

Edit `.github/workflows/daily-report.yml`:

```yaml
on:
  schedule:
    # Change cron schedule
    # Format: minute hour day month weekday
    - cron: '0 22 * * *'  # UTC 22:00 = KST 07:00

    # Examples:
    # '0 23 * * *'  # 08:00 KST
    # '0 0 * * *'   # 09:00 KST
    # '0 12 * * *'  # 21:00 KST
```

**Cron format reference:**
- `0 22 * * *` = Every day at 22:00 UTC (07:00 KST)
- `0 */6 * * *` = Every 6 hours
- `0 22 * * 1-5` = Weekdays only at 22:00 UTC

### Enable Audio Reports

Edit `config/config.yaml`:

```yaml
tts:
  enabled: true  # Change to true
  voice: ko-KR-SunHiNeural  # Korean female voice
  # voice: ko-KR-InJoonNeural  # Korean male voice
```

### Enable Telegram Notifications

1. Set up Telegram bot (see API Keys section above)
2. Add secrets to GitHub (see GitHub Setup above)
3. Edit `config/config.yaml`:

```yaml
publishing:
  telegram:
    enabled: true  # Change to true
    send_text: true
    send_audio: false  # Set to true if you want audio
```

---

## 4. Troubleshooting

### Issue: GitHub Actions fails with "API key not configured"

**Solution:**
- Double-check GitHub Secrets are set correctly
- Secret names must match exactly (case-sensitive)
- Re-enter secrets if necessary

### Issue: No news collected

**Solution:**
- Check NewsAPI key is valid
- Free tier has 100 requests/day limit
- Check config.yaml has valid sources

### Issue: Weather data not working

**Solution:**
- Verify OpenWeatherMap API key
- Check city name in config.yaml
- API key needs ~10 minutes to activate after creation

### Issue: AI summarization fails

**Solution:**
- Verify OpenAI API key is valid
- Check billing is set up on OpenAI account
- Try gpt-3.5-turbo (cheaper) instead of gpt-4

### Issue: GitHub Actions quota exceeded

**Solution:**
- Public repositories get 2,000 minutes/month free
- One daily run uses ~2-3 minutes
- Monitor usage: Settings → Billing → Plans and usage

---

## 5. Cost Estimate

**Monthly costs (approximate):**

| Service | Cost | Notes |
|---------|------|-------|
| NewsAPI | FREE | 100 requests/day free tier |
| OpenWeatherMap | FREE | 1,000 calls/day free tier |
| OpenAI GPT-4 | ~$0.30/month | ~$0.01 per report × 30 days |
| OpenAI GPT-3.5 | ~$0.03/month | Much cheaper alternative |
| GitHub Actions | FREE | 2,000 minutes/month for public repos |
| Telegram Bot | FREE | Unlimited |

**Total**: ~$0.30/month (or $0.03 with GPT-3.5-turbo)

---

## 6. Advanced Configuration

### Use Claude instead of GPT-4

```env
# In .env
AI_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=your_claude_key
```

### Multiple Daily Reports

Create multiple workflow files:

`.github/workflows/morning-report.yml`:
```yaml
name: Morning Report
on:
  schedule:
    - cron: '0 22 * * *'  # 07:00 KST
```

`.github/workflows/evening-report.yml`:
```yaml
name: Evening Report
on:
  schedule:
    - cron: '0 9 * * *'  # 18:00 KST
```

### Custom Report Template

Edit `src/generators/text_generator.py`:
- Modify `_generate_weather_section()` for weather format
- Modify `_generate_news_sections()` for news format
- Modify CSS in `generate_html()` for styling

---

## 7. Maintenance

### Update Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Update all packages
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Commit updated requirements
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### Monitor GitHub Actions

- Check Actions tab regularly
- Review logs if failures occur
- Check email for failure notifications

### Archive Old Reports

```bash
# Move old reports to archive
mv reports/daily/report_2025-*.md reports/archive/
git add reports/
git commit -m "Archive old reports"
git push
```

---

## 8. Next Steps

After successful setup:

1. ⭐ Star the repository on GitHub
2. 📖 Read through generated reports
3. 🔧 Customize config.yaml to your preferences
4. 📱 Set up Telegram notifications (optional)
5. 🎙️ Enable audio reports (optional)
6. 🤝 Share with others!

---

## Support

If you encounter issues:

1. Check GitHub Actions logs for errors
2. Review this setup guide
3. Check README.md for additional info
4. Search/create GitHub Issues

---

**Happy reporting! 📰**
