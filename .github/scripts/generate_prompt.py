import os
import json
import requests

class PRAnalyzer:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.pr_number = os.getenv('PR_NUMBER')
        self.repo_name = os.getenv('REPO_NAME')
        self.pr_title = os.getenv('PR_TITLE', '')
        self.pr_body = os.getenv('PR_BODY', '')
        
    def get_pr_changes(self):
        """Get PR file changes from GitHub API"""
        url = f"https://api.github.com/repos/{self.repo_name}/pulls/{self.pr_number}/files"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                files = response.json()
                return files
            else:
                print(f"❌ GitHub API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error fetching changes: {e}")
            return []
    
    def analyze_changes(self, files):
        """Analyze the file changes"""
        if not files:
            return "No file changes detected"
        
        summary = {
            'total_files': len(files),
            'files_added': len([f for f in files if f['status'] == 'added']),
            'files_modified': len([f for f in files if f['status'] == 'modified']),
            'files_deleted': len([f for f in files if f['status'] == 'removed']),
            'total_additions': sum(f.get('additions', 0) for f in files),
            'total_deletions': sum(f.get('deletions', 0) for f in files),
        }
        
        # Get file types
        file_types = {}
        for f in files:
            ext = f['filename'].split('.')[-1] if '.' in f['filename'] else 'no-ext'
            file_types[ext] = file_types.get(ext, 0) + 1
        
        return summary, file_types, files[:10]  # First 10 files for review

def main():
    analyzer = PRAnalyzer()
    
    print("=" * 60)
    print("🔍 PR DOCUMENTATION ANALYSIS")
    print("=" * 60)
    
    # Display PR info
    print(f"📋 PR #{analyzer.pr_number}")
    print(f"📝 Title: {analyzer.pr_title}")
    print(f"📄 Description: {analyzer.pr_body[:200]}{'...' if len(analyzer.pr_body) > 200 else ''}")
    print(f"🏢 Repository: {analyzer.repo_name}")
    
    print("\n" + "=" * 60)
    print("📁 FILE CHANGES")
    print("=" * 60)
    
    # Get and analyze changes
    files = analyzer.get_pr_changes()
    
    if files:
        summary, file_types, sample_files = analyzer.analyze_changes(files)
        
        print(f"📊 Summary:")
        print(f"   • Total files changed: {summary['total_files']}")
        print(f"   • Files added: {summary['files_added']}")
        print(f"   • Files modified: {summary['files_modified']}")
        print(f"   • Files deleted: {summary['files_deleted']}")
        print(f"   • Lines added: {summary['total_additions']}")
        print(f"   • Lines removed: {summary['total_deletions']}")
        
        print(f"\n📂 File types changed:")
        for ext, count in file_types.items():
            print(f"   • .{ext}: {count} files")
        
        print(f"\n📋 Files changed (first 10):")
        for f in sample_files:
            status_emoji = {"added": "➕", "modified": "✏️", "removed": "❌"}.get(f['status'], "📝")
            print(f"   {status_emoji} {f['filename']} (+{f.get('additions', 0)}/-{f.get('deletions', 0)})")
        
        print("\n" + "=" * 60)
        print("📝 DOCUMENTATION ASSESSMENT")
        print("=" * 60)
        
        # Simple rule-based assessment
        needs_docs = []
        
        # Check for common patterns that need docs
        for f in files:
            filename = f['filename'].lower()
            if any(pattern in filename for pattern in ['api', 'endpoint', 'route']):
                needs_docs.append(f"🔗 API changes detected in {f['filename']}")
            if any(pattern in filename for pattern in ['config', 'env', 'setting']):
                needs_docs.append(f"⚙️ Configuration changes in {f['filename']}")
            if any(pattern in filename for pattern in ['readme', 'doc', 'guide']):
                needs_docs.append(f"📚 Documentation file modified: {f['filename']}")
            if f['filename'].endswith('.sql') or 'migration' in filename:
                needs_docs.append(f"🗄️ Database changes in {f['filename']}")
        
        if needs_docs:
            print("⚠️ Documentation updates likely needed:")
            for item in needs_docs[:5]:  # Limit to 5 items
                print(f"   {item}")
        else:
            print("✅ No obvious documentation updates detected")
        
        # Simple priority assessment
        if summary['total_files'] > 10 or summary['total_additions'] > 200:
            priority = "🔴 HIGH"
        elif summary['total_files'] > 3 or summary['total_additions'] > 50:
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"
        
        print(f"\n📊 Documentation Priority: {priority}")
        
    else:
        print("❌ Could not retrieve file changes")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Save basic summary for next steps
    summary_data = {
        'pr_number': analyzer.pr_number,
        'pr_title': analyzer.pr_title,
        'repo_name': analyzer.repo_name,
        'files_changed': len(files),
        'timestamp': os.popen('date').read().strip()
    }
    
    with open('pr_analysis.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("💾 Analysis saved to pr_analysis.json")

if __name__ == "__main__":
    main()
