"""
Copy monitoring reports from Docker container to local directory
Workaround for Docker Desktop Windows path issues
"""
import subprocess
import os
import shutil

# Configuration
CONTAINER = "airflow_scheduler"
SOURCE_DIR = "/app/monitoring_reports"
DEST_DIR = "monitoring_reports"

# Files to copy
FILES = [
    "monitoring_report.html",
    "psi_over_time.png",
    "performance_dashboard.png",
    "feature_drift_heatmap.png",
    "data_quality_trends.png",
    "prediction_distribution.png"
]

print("="*80)
print("COPYING MONITORING REPORTS FROM DOCKER CONTAINER")
print("="*80)

# Create temp directory
temp_dir = "temp_reports"
os.makedirs(temp_dir, exist_ok=True)

success_count = 0
failed_files = []

for filename in FILES:
    source_path = f"{SOURCE_DIR}/{filename}"
    temp_path = os.path.join(temp_dir, filename)
    dest_path = os.path.join(DEST_DIR, filename)

    print(f"\nCopying {filename}...", end=" ")

    try:
        # Use docker exec to cat the file and save to temp
        cmd = f'docker exec {CONTAINER} cat {source_path}'
        result = subprocess.run(cmd, shell=True, capture_output=True)

        if result.returncode == 0:
            # Write to temp file
            with open(temp_path, 'wb') as f:
                f.write(result.stdout)

            # Move to destination
            shutil.copy2(temp_path, dest_path)

            file_size = os.path.getsize(dest_path)
            print(f"✓ ({file_size:,} bytes)")
            success_count += 1
        else:
            print(f"✗ (Error: {result.stderr.decode()[:50]})")
            failed_files.append(filename)

    except Exception as e:
        print(f"✗ (Exception: {str(e)[:50]})")
        failed_files.append(filename)

# Cleanup temp directory
shutil.rmtree(temp_dir, ignore_errors=True)

# Summary
print("\n" + "="*80)
print(f"COPY COMPLETE: {success_count}/{len(FILES)} files copied successfully")
print("="*80)

if failed_files:
    print(f"\n⚠ Failed files: {', '.join(failed_files)}")

if success_count > 0:
    print(f"\n✓ Files saved to: {os.path.abspath(DEST_DIR)}")

    # List the copied files
    print("\nCopied files:")
    for filename in FILES:
        path = os.path.join(DEST_DIR, filename)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            mtime = os.path.getmtime(path)
            from datetime import datetime
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  • {filename:40s} {size_kb:8.1f} KB  {mtime_str}")

    # Open HTML report
    html_path = os.path.join(DEST_DIR, "monitoring_report.html")
    if os.path.exists(html_path):
        print(f"\n🌐 Opening HTML report...")
        import webbrowser
        webbrowser.open(os.path.abspath(html_path))

print("\nDone!")
