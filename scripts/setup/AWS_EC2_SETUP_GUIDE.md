# WebArena AWS EC2 Setup Guide (RECOMMENDED METHOD)

## Why Use AWS AMI Instead of Local Setup?

The WebArena team **strongly recommends** using their pre-configured AWS AMI because:
- ✅ Everything is pre-installed and configured
- ✅ Requires significant compute resources (4 vCPUs, 16GB RAM)
- ✅ All services are tested and working together
- ✅ Avoids the complex Docker networking issues you're experiencing locally
- ✅ Takes ~10 minutes to set up vs hours of debugging locally

**Cost Estimate**: t3a.xlarge costs ~$0.15/hour (~$3.60/day if running 24/7)

---

## Prerequisites

1. **AWS Account** - If you don't have one:
   - Go to https://aws.amazon.com/
   - Click "Create an AWS Account"
   - Follow the signup process (requires credit card)
   - AWS has a free tier, but this instance type is not covered

2. **Basic AWS Knowledge** (optional but helpful):
   - Understanding of EC2 instances
   - SSH basics

---

## Step-by-Step Setup Instructions

### Step 1: Access AWS EC2 Console

1. Log into AWS Console: https://console.aws.amazon.com/
2. **IMPORTANT**: Switch region to **us-east-2 (Ohio)** 
   - Look at top-right corner of AWS console
   - Click the region dropdown
   - Select "US East (Ohio) us-east-2"
   - ⚠️ The AMI is only available in us-east-2!

### Step 2: Create Security Group

1. In EC2 console, go to **Security Groups** (left sidebar under "Network & Security")
2. Click **"Create security group"**
3. Configure:
   - **Name**: `webarena-security-group`
   - **Description**: `Security group for WebArena services`
   - **VPC**: (leave default)

4. **Add Inbound Rules** - Click "Add rule" for each:

   | Type       | Protocol | Port Range | Source      | Description           |
   |------------|----------|------------|-------------|-----------------------|
   | SSH        | TCP      | 22         | My IP       | SSH access            |
   | HTTP       | TCP      | 80         | 0.0.0.0/0   | HTTP                  |
   | Custom TCP | TCP      | 3000       | 0.0.0.0/0   | Map service           |
   | Custom TCP | TCP      | 7770       | 0.0.0.0/0   | Shopping site         |
   | Custom TCP | TCP      | 7780       | 0.0.0.0/0   | Shopping admin        |
   | Custom TCP | TCP      | 8023       | 0.0.0.0/0   | GitLab                |
   | Custom TCP | TCP      | 8888       | 0.0.0.0/0   | Wikipedia             |
   | Custom TCP | TCP      | 9999       | 0.0.0.0/0   | Forum (Reddit-like)   |
   | Custom TCP | TCP      | 4399       | 0.0.0.0/0   | Homepage              |

5. Click **"Create security group"**

### Step 3: Create SSH Key Pair (if you don't have one)

1. In EC2 console, go to **Key Pairs** (left sidebar under "Network & Security")
2. Click **"Create key pair"**
3. Configure:
   - **Name**: `webarena-key`
   - **Key pair type**: RSA
   - **Private key format**: `.pem` (for Mac/Linux) or `.ppk` (for PuTTY on Windows)
4. Click **"Create key pair"** - it will download automatically
5. **IMPORTANT**: Move the key to a safe location and set permissions:
   ```bash
   mkdir -p ~/.ssh
   mv ~/Downloads/webarena-key.pem ~/.ssh/
   chmod 400 ~/.ssh/webarena-key.pem
   ```

### Step 4: Find the WebArena AMI

1. In EC2 console, go to **AMIs** (left sidebar under "Images")
2. Change filter from "Owned by me" to **"Public images"**
3. Search for: `ami-08a862bf98e3bd7aa`
4. You should see: **webarena-with-configurable-map-backend**
5. Select it and click **"Launch instance from AMI"**

### Step 5: Launch EC2 Instance

Configure the instance:

1. **Name**: `webarena-server`

2. **Instance type**: `t3a.xlarge`
   - 4 vCPUs, 16 GB RAM
   - ⚠️ This is NOT free tier eligible (~$0.15/hour)

3. **Key pair**: Select `webarena-key` (or your existing key)

4. **Network settings**:
   - Click "Edit"
   - **Firewall (security groups)**: Select existing security group
   - Choose `webarena-security-group` (created in Step 2)

5. **Configure storage**:
   - **Size**: `1000 GiB` (1 TB)
   - **Volume type**: gp3 (recommended)
   - ⚠️ This is important - services need this space!

6. **Advanced details** (expand this section):
   - Scroll down to **User data**
   - **RECOMMENDED**: Leave this **EMPTY** for now
   - The map will work with the default AWS infrastructure
   - ⚠️ Only set `MAP_BACKEND_IP=YOUR_IP` if you've set up a separate map backend server (see Optional Setup below)

7. **Review and Launch**:
   - Review the Summary on the right
   - Click **"Launch instance"**

8. Wait for instance to start (2-3 minutes)
   - Go to **Instances** in left sidebar
   - Wait for "Instance state" to show "Running"
   - Wait for "Status check" to show "2/2 checks passed"

### Step 6: Get Your Instance's Public Address

You have two options for accessing your instance:

#### **Option A: Use Instance's Public DNS (Recommended - No Elastic IP needed)**

1. Go to **EC2 → Instances**
2. Select your `webarena-server` instance
3. Copy the **Public IPv4 DNS** from the details panel
   - It looks like: `ec2-3-150-230-27.us-east-2.compute.amazonaws.com`
4. **Save this hostname** - you'll need it for configuration!

**Note:** This DNS name will work as long as the instance is running. The IP will change if you **stop and start** the instance (but NOT on reboot).

#### **Option B: Create Elastic IP (Only if you need a permanent IP)**

⚠️ **AWS Limit:** Free tier accounts have a limit of 5 Elastic IPs per region. If you get an error, either:
- Release unused Elastic IPs in EC2 → Elastic IPs
- Use Option A instead
- Request a limit increase via Service Quotas

1. In EC2 console, go to **Elastic IPs** (left sidebar under "Network & Security")
2. Click **"Allocate Elastic IP address"**
3. Click **"Allocate"**
4. Select the newly created Elastic IP
5. Click **Actions** → **Associate Elastic IP address**
6. Configure:
   - **Instance**: Select your `webarena-server`
   - **Private IP**: (leave default)
7. Click **"Associate"**
8. **Note the Public DNS** - it will look like: `ec2-3-150-230-27.us-east-2.compute.amazonaws.com`

### Step 7: Connect to Your Instance

From your terminal:

```bash
# Your actual hostname
export WEBARENA_HOST="ec2-3-150-230-27.us-east-2.compute.amazonaws.com"

# SSH into the instance
ssh -i ~/.ssh/webarena-key.pem ubuntu@$WEBARENA_HOST
```

### Step 8: Start WebArena Services

Once connected via SSH, run these commands:

```bash
# Start all Docker containers
docker start gitlab
docker start shopping
docker start shopping_admin
docker start forum
docker start kiwix33

# Start the map service
cd /home/ubuntu/openstreetmap-website/
docker compose start
```

Wait about 1-2 minutes for services to start.

### Step 9: Configure Services

⚠️ **CRITICAL**: Replace `<your-server-hostname>` with your actual hostname from Step 6!

```bash
# Your actual hostname
export HOSTNAME="ec2-3-150-230-27.us-east-2.compute.amazonaws.com"

# Configure shopping site
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOSTNAME}:7770"
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value=\"http://${HOSTNAME}:7770/\" WHERE path = \"web/secure/base_url\";"
docker exec shopping /var/www/magento2/bin/magento cache:flush

# Configure shopping admin
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOSTNAME}:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value=\"http://${HOSTNAME}:7780/\" WHERE path = \"web/secure/base_url\";"
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

# Configure GitLab
docker exec gitlab sed -i "s|^external_url.*|external_url 'http://${HOSTNAME}:8023'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure
```

**If GitLab shows 502 errors**, run:
```bash
docker exec gitlab rm -f /var/opt/gitlab/postgresql/data/postmaster.pid
docker exec gitlab /opt/gitlab/embedded/bin/pg_resetwal -f /var/opt/gitlab/postgresql/data
docker exec gitlab gitlab-ctl restart
```

### Step 10: Test All Services

```bash
# Still on the EC2 instance, run:
echo "Testing all services..."
curl -s -o /dev/null -w "Shopping (7770): %{http_code}\n" http://$HOSTNAME:7770
curl -s -o /dev/null -w "Shopping Admin (7780): %{http_code}\n" http://$HOSTNAME:7780
curl -s -o /dev/null -w "Forum (9999): %{http_code}\n" http://$HOSTNAME:9999
curl -s -o /dev/null -w "Wikipedia (8888): %{http_code}\n" http://$HOSTNAME:8888
curl -s -o /dev/null -w "Map (3000): %{http_code}\n" http://$HOSTNAME:3000
curl -s -o /dev/null -w "GitLab (8023): %{http_code}\n" http://$HOSTNAME:8023
curl -s -o /dev/null -w "Map tile: %{http_code}\n" http://$HOSTNAME:3000/tile/0/0/0.png
curl -s -o /dev/null -w "Homepage (4399): %{http_code}\n" http://$HOSTNAME:4399
```

✅ All should return **200** (might take a minute or two after configuration)

### Step 11: Test from Your Browser

Open these URLs in your browser:
- Shopping: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7770
- Shopping Admin: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7780
- Forum: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:9999
- Wikipedia: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8888
- Map: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:3000
- GitLab: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8023
- Homepage: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:4399

---

## Step 12: Set Up the Homepage

The homepage lists all available websites which the agent can use to navigate to different sites. This is **required** for the agent to work properly.

**On your EC2 instance** (via SSH), run these commands:

```bash
# Your actual hostname (same as before)
export HOSTNAME="ec2-3-150-230-27.us-east-2.compute.amazonaws.com"

# Navigate to the homepage directory
cd /home/ubuntu/webarena-homepage

# Update the HTML template with your hostname
perl -pi -e "s|<your-server-hostname>|${HOSTNAME}|g" templates/index.html

# Start the Flask server in the background using tmux
tmux new-session -d -s homepage "flask run --host=0.0.0.0 --port=4399"

# Verify it's running
curl -s -o /dev/null -w "Homepage: %{http_code}\n" http://localhost:4399
```

✅ Should return **Homepage: 200**

**Test the homepage** in your browser:
- Homepage: http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:4399

You should see a page listing all the WebArena services with working links.

**To keep the homepage running after logout:**

The `tmux` session will keep the Flask server running. To check on it:
```bash
# View the homepage tmux session
tmux attach -t homepage

# Detach from tmux (keep it running): Press Ctrl+B, then D
```

**To restart homepage after instance reboot:**
```bash
# SSH into instance, then:
cd /home/ubuntu/webarena-homepage
tmux new-session -d -s homepage "flask run --host=0.0.0.0 --port=4399"
```

---

## Step 13: Update Your Local Configuration

On your **local machine** (where you run ReasoningBank), update the environment variables:

```bash
# Edit your config or environment file
export SHOPPING="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7770"
export SHOPPING_ADMIN="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7780"
export REDDIT="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:9999"
export GITLAB="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8023"
export MAP="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:3000"
export WIKIPEDIA="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8888"
export HOMEPAGE="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:4399"
```

Or create a file `.env.webarena`:

```bash
cat > ~/.env.webarena << 'EOF'
export SHOPPING="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7770"
export SHOPPING_ADMIN="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7780"
export REDDIT="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:9999"
export GITLAB="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8023"
export MAP="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:3000"
export WIKIPEDIA="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:8888"
export HOMEPAGE="http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:4399"
EOF

# Then source it before running experiments
source ~/.env.webarena
```

---

## Managing Your Instance

### Stop Instance (to save money when not using)
```bash
# From AWS Console: Select instance → Instance state → Stop instance
# Or via CLI:
aws ec2 stop-instances --instance-ids i-xxxxx --region us-east-2
```

### Start Instance Again
```bash
# From AWS Console: Select instance → Instance state → Start instance
# Then SSH in and run the docker start commands from Step 8
```

### Terminate Instance (permanent deletion)
⚠️ **This will DELETE everything!** Only do this when completely done.
```bash
# From AWS Console: Select instance → Instance state → Terminate instance
```

---

## Troubleshooting

### Can't find the AMI?
- Make sure you're in **us-east-2 (Ohio)** region
- Change AMI filter to "Public images"
- Search by AMI ID: `ami-08a862bf98e3bd7aa`

### Can't connect via SSH?
- Check security group allows SSH (port 22) from your IP
- Verify key permissions: `chmod 400 ~/.ssh/webarena-key.pem`
- Check instance is "Running" with "2/2 status checks passed"

### Services return 502/503 errors?
- Wait 2-3 minutes after starting - services need time to boot
- Check if containers are running: `docker ps`
- Restart problematic service: `docker restart <service-name>`

### GitLab shows 502?
- See the GitLab fix commands in Step 9
- GitLab takes 3-5 minutes to fully start

---

## Cost Management Tips

1. **Stop when not using**: Stop instance when not running experiments
2. **Use Spot Instances**: Can save 70% but instance may be interrupted
3. **Set billing alerts**: AWS Console → Billing → Budgets
4. **Remember to terminate**: When project is done, terminate to avoid charges

---

## Next Steps

Once your WebArena environment is running on EC2:

1. ✅ Test all services work from your browser
2. ✅ Update your local ReasoningBank configuration
3. ✅ Run a simple test: `python reproduce_table1.py --model gemini-2.0-flash --max-tasks 1`
4. ✅ If test passes, run full evaluation!

---

## Summary: Why This Approach is Better

| Aspect | Local Setup | AWS AMI Setup |
|--------|-------------|---------------|
| Setup Time | Hours (with debugging) | 20-30 minutes |
| Reliability | Port conflicts, resource issues | Pre-configured, tested |
| Resources | Requires powerful local machine | Dedicated cloud resources |
| Networking | Complex localhost routing | Simple public IPs |
| Cost | Free but frustrating | ~$3.60/day but reliable |

**Bottom line**: Spend $10-20 on AWS and get your research done, rather than spending days debugging local Docker issues! 🚀

---

## Optional: Setting Up Your Own Map Backend

⚠️ **You probably don't need this!** The map works fine with default infrastructure.

Only set up your own map backend if you need:
- Custom map tile data
- Full control over geocoding/routing services
- To avoid dependencies on WebArena's default infrastructure

### Steps to Set Up Custom Map Backend:

1. **Launch a SECOND EC2 instance** (in addition to your main WebArena instance):
   - Region: us-east-2 (Ohio)
   - AMI: Ubuntu 24.04 LTS
   - Instance type: t3a.xlarge
   - Storage: 1000GB
   - Security group: Same as your WebArena instance

2. **Get the setup script**:
   ```bash
   # On your local machine
   cd /Users/hivamoh/Desktop/ReasoningBank/data/webarena_repo
   cat webarena-map-backend-boot-init.yaml
   ```

3. **During instance launch**:
   - Copy the entire contents of `webarena-map-backend-boot-init.yaml`
   - Paste into **User Data** field
   - Launch the instance

4. **Wait for setup** (60-90 minutes):
   - The script will automatically download ~180GB of map data
   - Set up tile server, geocoding server, and routing server
   - You can monitor progress by SSH'ing in and checking: `tail -f /var/log/cloud-init-output.log`

5. **Get the backend IP**:
   - Go to EC2 console
   - Select your map backend instance
   - Note the **Public IPv4 address** (e.g., `18.XX.XX.XX`)

6. **Configure your WebArena instance**:
   - When launching your main WebArena instance (Step 5 above)
   - In **User Data**, enter:
     ```
     MAP_BACKEND_IP=18.XX.XX.XX
     ```
   - Replace with your actual backend IP

### Cost for Custom Map Backend:
- Additional t3a.xlarge: ~$0.15/hour (~$3.60/day)
- Storage (1TB): ~$0.10/GB/month (~$100/month)
- **Total: ~$107/month** if running 24/7

**Again, this is optional!** For most research purposes, the default map infrastructure works perfectly fine.

