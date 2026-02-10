# MuSc Industrial Anomaly Detection - Quick Start Guide

## What This Tool Does

The MuSc Industrial Anomaly Detection system automatically detects defects, scratches, and anomalies in your production line using AI. No training required - it works right out of the box!

**Key Features:**
- Real-time defect detection from camera feeds
- Automatic highlighting of problem areas
- Works with USB cameras and IP cameras
- Analyzes live video or pre-recorded footage
- No programming knowledge needed

---

## First Time Setup (5 Minutes)

### Step 1: Connect Your Camera

**For USB Cameras:**
1. Plug in your USB camera
2. In the "Camera Source" section, use the default device index (0)
3. Click "Connect USB Camera"
4. You should see "Status: Connected to USB Camera" in green

**For IP Cameras:**
1. Find your camera's URL (check camera documentation)
2. Enter the URL in the format: `rtsp://username:password@192.168.1.100:554/stream`
3. Click "Connect IP Camera"
4. You should see "Status: Connected to IP Camera" in green

**Need Help?** Hover your mouse over any field to see helpful tips!

### Step 2: Choose Your Settings (Optional)

The default settings work for most cases, but you can adjust them:

1. Click the **Configuration** button
2. Look at these key settings:

   **AI Model:** (default is fine)
   - Leave as "dinov2_vitb14" for balanced performance
   - Change to "dino_deitsmall16" if your computer is slow
   - Change to "dinov2_vitl14" for maximum accuracy (requires powerful GPU)

   **Detection Sensitivity:** (90% recommended)
   - High (80-100%): Only flags obvious defects - fewer false alarms
   - Medium (60-80%): Balanced - catches most issues
   - Low (0-60%): Very sensitive - catches subtle defects but may have false alarms

   **Heatmap Intensity:** (35% recommended)
   - Controls how much of the defect area is highlighted in red

3. Click **OK** to save

### Step 3: Start Detecting Defects

1. Make sure your camera is showing live video
2. Adjust **Duration** (default 3 seconds is good)
   - This is how long to record before analyzing
3. Adjust **Live Target FPS** (15 recommended)
   - Higher = smoother but slower
4. Click **Start** button
5. Wait for frames to collect (progress bar fills up)
6. Analysis happens automatically
7. Check the **Defect Level** display:
   - 0-30%: Normal (no defects)
   - 30-70%: Minor issues
   - 70-90%: Moderate defects
   - 90-100%: Major defects

**Tip:** Check the "Continuous" box to keep analyzing automatically!

---

## Understanding the Interface

### Main Display
- **Camera Feed**: Shows live video from your camera
- **Defect Level**: Shows the severity of detected issues (0-100%)
- **Analysis Time**: How long it took to process the images

### Status Bar (Bottom of Window)
Always shows what the system is doing:
- **Left**: Current operation ("Ready to start", "Analyzing for defects...", etc.)
- **Center**: Which AI model is loaded
- **Right**: Camera connection status

### Tabs on the Right Side

**1. Detected Defects Tab**
- Shows images where defects were found
- Red highlights show problem areas
- Click "Show Defect Highlights" to toggle the red overlay on/off
- **Load More** button: Shows all detected defects (not just recent 10)

**2. Captured Images Tab**
- Images you manually captured using the "Capture" button
- Use this to save examples for later review

**3. Reference Library Tab**
- Store examples of GOOD (defect-free) products here
- Helps the AI learn what's normal for your production line
- To add: Select images in other tabs → Click "Save to Reference Library"

---

## Common Tasks

### Adjusting Sensitivity
**Too many false alarms?**
1. Click **Configuration**
2. Increase "Detection Sensitivity" to 90-95%
3. Click **OK**

**Missing real defects?**
1. Click **Configuration**
2. Decrease "Detection Sensitivity" to 70-80%
3. Click **OK**

### Saving Defect Images
1. Go to **Detected Defects** tab
2. Check the boxes next to images you want to save
3. Click **Save Selected**
4. Choose a folder on your computer

### Analyzing a Video File
1. Switch to **Load Video** tab
2. Click **Browse Video**
3. Select your video file (.mp4, .avi, .mov)
4. Click **Play** to preview
5. Click **Start Analysis** to begin detecting defects
6. Defects will appear in the **Detected Defects** tab

### Analyzing a Folder of Images
1. Switch to **Load Video** tab
2. Click **Browse Folder**
3. Select folder containing images (.png, .jpg, .bmp)
4. Click **Analyze Folder Images**
5. Results appear in **Detected Defects** tab

### Building a Reference Library
To improve detection accuracy, add examples of good products:
1. Capture or select images of defect-free products
2. Check the boxes next to those images
3. Click **Save to Reference Library**
4. The AI will use these as examples of "normal"

---

## Troubleshooting

### Camera Won't Connect
**Try these:**
- Make sure camera is plugged in and powered on
- Try a different USB port
- Try increasing the device index (0, 1, 2, etc.)
- Close other apps that might be using the camera
- For IP cameras: verify the IP address and network connection

### Running Very Slow
**Solutions:**
1. You might be in CPU mode (no GPU detected)
   - GPU is required for real-time detection
   - Check status bar: Does it mention GPU or CPU?
2. Try a faster model:
   - Click **Configuration**
   - Change AI Model to "dino_deitsmall16"
   - Click **OK**
3. Reduce image size:
   - Click **Configuration**
   - Change "Image Size" to 224 (if not already)
   - Click **OK**

### No Defects Detected (But You See Them)
**The AI is not sensitive enough:**
1. Click **Configuration**
2. Lower "Detection Sensitivity" to 60-70%
3. Click **OK**
4. Try analyzing again

### Too Many False Alarms
**The AI is too sensitive:**
1. Click **Configuration**
2. Increase "Detection Sensitivity" to 90-95%
3. Click **OK**
4. Add good examples to Reference Library

### Can't See Red Highlights on Defects
1. Go to **Detected Defects** tab
2. Make sure "Show Defect Highlights" is checked
3. If still not visible, try adjusting Heatmap Intensity:
   - Click **Configuration**
   - Lower "Heatmap Intensity" to 20-30%
   - Click **OK**

---

## Tips for Best Results

### 1. Lighting Matters
- Consistent, even lighting works best
- Avoid shadows and glare
- Same lighting conditions during normal production

### 2. Build a Good Reference Library
- Add 10-20 examples of perfect products
- Use images from the same camera angle
- Include variety of normal conditions
- Update when production changes

### 3. Camera Positioning
- Keep camera steady (mount it if possible)
- Frame your product consistently
- Same distance and angle each time
- Fill the frame with your product

### 4. Start Conservative
- Begin with higher Detection Sensitivity (90%)
- Gradually lower if you're missing defects
- Better to start strict and adjust down

### 5. Monitor Performance
- Watch the Analysis Time
- Should complete before next batch
- If too slow, reduce Duration or FPS

### 6. Use Continuous Mode for Production
- Check "Continuous" checkbox
- System will keep monitoring automatically
- Good for real-time quality control

---

## Understanding Defect Levels

The system shows three scores:
- **Max**: Worst defect found in the batch
- **Min**: Best quality item in the batch
- **Avg**: Average quality across all items

**What the percentages mean:**
- **0-30%**: ✓ Normal - no significant issues
- **30-50%**: ⚠ Minor - small scratches, dirt, minor cosmetic issues
- **50-70%**: ⚠ Moderate - visible defects worth inspecting
- **70-90%**: ⚠️ Significant - clear quality problems
- **90-100%**: ❌ Major - severe defects, reject immediately

**Important:** These are guidelines. The right threshold depends on your specific product and quality requirements.

---

## Getting Help

### Built-in Help
- **Hover over any control** to see what it does
- **Status bar** shows what's currently happening
- **Color coding**: Green = good, Orange = warning, Red = error, Blue = working

### Need More Help?
- Check the README.md file for technical details
- Report issues on GitHub: [your-repo-url]
- Email support: [your-email]

---

## Quick Reference Card

### Essential Buttons
| Button | What It Does |
|--------|-------------|
| **Start** | Begin collecting and analyzing frames |
| **Stop** | Stop analysis |
| **Capture** | Save a single frame |
| **Configuration** | Adjust AI model and sensitivity |

### Key Settings
| Setting | Recommended Value | Purpose |
|---------|-------------------|---------|
| **Duration** | 3 seconds | How long to record before analyzing |
| **Live Target FPS** | 15 | Frames per second (higher = slower) |
| **Detection Sensitivity** | 90% | How strict to be (higher = fewer false alarms) |
| **Heatmap Intensity** | 35% | How much defect area to highlight |

### Status Colors
- 🟢 **Green**: Ready, connected, or complete
- 🟠 **Orange**: Warning or not connected
- 🔵 **Blue**: Working (analyzing or collecting)
- 🔴 **Red**: Error or problem

---

## Workflow Examples

### Example 1: Live Quality Control Station
1. Mount USB camera over production line
2. Connect camera (device 0)
3. Set Detection Sensitivity to 90%
4. Set Duration to 2 seconds, FPS to 15
5. Check "Continuous" checkbox
6. Click Start
7. System monitors continuously
8. Check Detected Defects tab periodically
9. Review and save any flagged items

### Example 2: Batch Inspection of Parts
1. Place parts in front of camera one by one
2. Click "Capture" for each part
3. After collecting 20-30 images:
   - Go to Captured Images tab
   - Select all images
   - Click "Analyze" (via context menu)
4. Review results in Detected Defects tab
5. Sort good vs. bad parts

### Example 3: Analyzing Pre-Recorded Video
1. Switch to Load Video tab
2. Browse and select your video file
3. Set Duration to 3 seconds, FPS to 30
4. Click "Start Analysis"
5. Let it process the entire video
6. Review all detected defects in sidebar
7. Export defect images for reporting

---

## Best Practices Checklist

Before starting production monitoring:
- ✓ Camera mounted securely
- ✓ Lighting consistent and even
- ✓ Camera focused on product
- ✓ Test with sample parts (both good and defective)
- ✓ Detection Sensitivity tuned to your needs
- ✓ Reference Library contains 10+ good examples
- ✓ Continuous mode enabled
- ✓ Monitor analysis time < duration time

---

## Frequently Asked Questions

**Q: Do I need to train the AI?**
A: No! The system works out of the box using pre-trained models. However, adding good examples to the Reference Library can improve accuracy for your specific products.

**Q: How fast can it detect defects?**
A: Depends on your hardware. With a modern GPU (RTX 3060+), you can analyze ~15-20 frames per second. Without a GPU, it will be much slower (10-30x).

**Q: Can it work without an internet connection?**
A: Yes! After the AI model downloads (first time only), everything runs locally. No internet required.

**Q: What types of defects can it detect?**
A: It's designed for visual anomalies: scratches, dents, missing parts, discoloration, contamination, surface defects, assembly errors, etc.

**Q: How accurate is it?**
A: Accuracy depends on your specific use case, but typically 90%+ for obvious defects. Fine-tune with Detection Sensitivity and Reference Library for best results.

**Q: Can I use multiple cameras?**
A: You can use one camera at a time. To switch cameras, disconnect the current one and connect a different one.

---

## Your First Session Checklist

Follow this for your first session:

1. ☐ Connect camera and verify green status
2. ☐ Click Configuration and review settings
3. ☐ Click Start and let it analyze a few batches
4. ☐ Check Defect Level percentages
5. ☐ Look at Detected Defects tab - does it make sense?
6. ☐ Adjust Detection Sensitivity if needed
7. ☐ Capture some good examples and add to Reference Library
8. ☐ Try Continuous mode
9. ☐ Export some defect images to a folder

**Congratulations!** You're now ready to use MuSc for industrial anomaly detection.

---

*For technical documentation, see README.md*
*For implementation details, see PHASE1_IMPROVEMENTS.md*
