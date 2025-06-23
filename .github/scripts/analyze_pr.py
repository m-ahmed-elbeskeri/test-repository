import os
import json
import openai
from typing import List, Dict
import requests

class PRAnalyzer:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.pr_number = os.getenv('PR_NUMBER')
        self.repo_name = os.getenv('REPO_NAME')
        
    def get_file_changes(self) -> List[Dict]:
        """Get detailed file changes from GitHub API"""
        url = f"https://api.github.com/repos/{self.repo_name}/pulls/{self.pr_number}/files"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return []
    
    def analyze_code_changes(self, file_changes: List[Dict]) -> str:
        """Analyze code changes to understand impact"""
        changes_summary = []
        
        for file in file_changes:
            filename = file['filename']
            status = file['status']  # added, modified, removed
            additions = file.get('additions', 0)
            deletions = file.get('deletions', 0)
            
            # Get patch content if available
            patch = file.get('patch', '')
            
            changes_summary.append({
                'file': filename,
                'status': status,
                'additions': additions,
                'deletions': deletions,
                'patch_preview': patch[:500] if patch else ''  # First 500 chars
            })
            
        return json.dumps(changes_summary, indent=2)
    
    def generate_documentation_outline(self, pr_data: Dict, code_changes: str) -> Dict:
        """Use LLM to generate documentation update outline"""
        
        prompt = f"""
        Analyze this Pull Request and generate a comprehensive outline of documentation updates needed.
        
        PR Title: {pr_data.get('title', '')}
        PR Description: {pr_data.get('body', '')}
        
        Code Changes Summary:
        {code_changes}
        
        Based on these changes, create a structured outline of documentation updates needed. Consider:
        1. What new features/APIs need documenting
        2. What existing docs need updating
        3. What architectural changes require documentation
        4. What configuration/setup changes need docs
        5. What breaking changes need migration guides
        
        Respond in this JSON format:
        {{
            "summary": "Brief overview of documentation impact",
            "priority": "high|medium|low",
            "categories": {{
                "new_documentation": [
                    {{
                        "title": "Document title",
                        "description": "What needs to be documented",
                        "confluence_space": "Suggested space",
                        "urgency": "high|medium|low"
                    }}
                ],
                "updates_needed": [
                    {{
                        "existing_page": "Page that needs updating",
                        "changes_required": "What changes are needed",
                        "confluence_space": "Space where page exists",
                        "urgency": "high|medium|low"
                    }}
                ],
                "breaking_changes": [
                    {{
                        "change": "Description of breaking change",
                        "migration_guide_needed": true/false,
                        "affected_docs": ["List of docs that need updates"]
                    }}
                ]
            }},
            "estimated_effort": "Small|Medium|Large",
            "recommended_actions": ["List of specific actions to take"]
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert who analyzes code changes and determines documentation requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Parse JSON response
            outline_text = response.choices[0].message.content
            return json.loads(outline_text)
            
        except Exception as e:
            print(f"Error generating outline: {e}")
            return {
                "summary": "Error analyzing PR - manual review needed",
                "priority": "medium",
                "error": str(e)
            }
    
    def save_outline(self, outline: Dict):
        """Save outline to file for next steps"""
        with open('documentation_outline.json', 'w') as f:
            json.dump(outline, f, indent=2)
        
        # Also create a readable summary
        summary = f"""
# Documentation Review Required

## Summary
{outline.get('summary', 'No summary available')}

**Priority:** {outline.get('priority', 'Unknown')}
**Estimated Effort:** {outline.get('estimated_effort', 'Unknown')}

## Changes Needed

### New Documentation Required
"""
        
        for item in outline.get('categories', {}).get('new_documentation', []):
            summary += f"- **{item.get('title', 'Untitled')}** ({item.get('urgency', 'medium')} priority)\n"
            summary += f"  - {item.get('description', 'No description')}\n"
            summary += f"  - Suggested space: {item.get('confluence_space', 'TBD')}\n\n"
        
        summary += "### Updates to Existing Documentation\n"
        for item in outline.get('categories', {}).get('updates_needed', []):
            summary += f"- **{item.get('existing_page', 'Unknown page')}** ({item.get('urgency', 'medium')} priority)\n"
            summary += f"  - Changes: {item.get('changes_required', 'No details')}\n\n"
        
        summary += "### Breaking Changes\n"
        for item in outline.get('categories', {}).get('breaking_changes', []):
            summary += f"- {item.get('change', 'Unknown change')}\n"
            if item.get('migration_guide_needed'):
                summary += "  - ⚠️ Migration guide required\n"
        
        summary += f"\n## Recommended Actions\n"
        for action in outline.get('recommended_actions', []):
            summary += f"- {action}\n"
        
        with open('documentation_summary.md', 'w') as f:
            f.write(summary)
        
        print("Documentation outline generated!")
        print(summary)

def main():
    analyzer = PRAnalyzer()
    
    # Get PR data from environment
    pr_data = {
        'title': os.getenv('PR_TITLE', ''),
        'body': os.getenv('PR_BODY', '')
    }
    
    # Get detailed file changes
    file_changes = analyzer.get_file_changes()
    code_changes = analyzer.analyze_code_changes(file_changes)
    
    # Generate outline
    outline = analyzer.generate_documentation_outline(pr_data, code_changes)
    
    # Save results
    analyzer.save_outline(outline)

if __name__ == "__main__":
    main()
