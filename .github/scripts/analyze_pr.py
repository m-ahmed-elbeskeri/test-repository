from __future__ import annotations as _annotations

import os
import json
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import httpx
from devtools import debug
from atlassian import Confluence
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# ============================================================================
# ASYNC-COMPATIBLE CONFLUENCE TOOLKIT
# ============================================================================

class ConfluenceToolkit:
    """
    An async-compatible wrapper for the synchronous atlassian-python-api library.
    It uses asyncio.to_thread to avoid blocking the event loop.
    """
    def __init__(self, url: str, username: str, api_token: str):
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )
    
    async def get_confluence_spaces(self) -> List[Dict]:
        """Get all Confluence spaces"""
        spaces = await asyncio.to_thread(self.confluence.get_all_spaces, start=0, limit=50)
        return spaces.get('results', [])
    
    async def search_confluence_using_cql(self, cql: str) -> List[Dict]:
        """Search Confluence using CQL"""
        results = await asyncio.to_thread(self.confluence.cql, cql)
        return results.get('results', [])
    
    async def get_confluence_page(self, page_id: str) -> Dict:
        """Get a specific Confluence page"""
        page = await asyncio.to_thread(
            self.confluence.get_page_by_id,
            page_id, 
            expand='body.storage,space,version'
        )
        return page
    
    async def get_pages_in_confluence_space(self, space_key: str) -> List[Dict]:
        """Get all pages in a Confluence space"""
        pages = await asyncio.to_thread(
            self.confluence.get_all_pages_from_space,
            space=space_key, 
            start=0, 
            limit=25
        )
        return pages

# ============================================================================
# DEPENDENCIES AND OUTPUT MODELS
# ============================================================================

@dataclass
class AnalysisDependencies:
    """Dependencies passed to agent tools and functions"""
    client: httpx.AsyncClient
    confluence_toolkit: ConfluenceToolkit
    pr_number: str
    repo_name: str
    pr_title: str
    pr_body: str
    file_analysis: Dict

class ConfluenceAction(BaseModel):
    """A single Confluence documentation action"""
    action: str = Field(description="Type of action: update_page, create_page, or review_page")
    page_id: Optional[str] = Field(description="Confluence page ID if updating existing page")
    space_key: str = Field(description="Confluence space key")
    page_title: str = Field(description="Title of the page to update or create")
    reason: str = Field(description="Why this documentation needs updating")
    priority: str = Field(description="Priority level: high, medium, or low")
    specific_changes: str = Field(description="Specific changes or additions needed")
    existing_content_summary: Optional[str] = Field(description="Brief summary of current page content if applicable")

class ConfluenceAnalysisResult(BaseModel):
    """Final structured output of the Confluence analysis"""
    confluence_actions: List[ConfluenceAction] = Field(description="List of documentation actions needed")
    summary: str = Field(description="Brief summary of documentation impact and rationale")
    total_actions: int = Field(description="Total number of actions identified")
    estimated_effort: str = Field(description="Estimated effort: Low, Medium, or High")
    spaces_affected: List[str] = Field(description="List of Confluence spaces that will be affected")

# ============================================================================
# PYDANTICAI AGENT WITH TOOLS
# ============================================================================

provider = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
model = OpenAIModel('gpt-4.1', provider=provider)

confluence_agent = Agent[AnalysisDependencies, ConfluenceAnalysisResult](
    model,
    deps_type=AnalysisDependencies,
    output_type=ConfluenceAnalysisResult,
    retries=3,  # Add retries for resilience
    system_prompt="""You are an elite documentation strategist responsible for analyzing pull requests and identifying necessary Confluence documentation updates. Your goal is to ensure that all code changes are properly reflected in the team's documentation.

Given a pull request with its context and file changes, you must:

1. **Analyze the Impact**: Understand what the code changes mean for existing documentation
2. **Search Confluence**: Use available tools to find relevant existing documentation
3. **Identify Actions**: Determine what documentation needs to be updated, created, or reviewed
4. **Prioritize**: Assign appropriate priority levels based on the impact and urgency
5. **Provide Specifics**: Give detailed recommendations for what changes need to be made

**Available Tools:**
- get_confluence_spaces: Get all available Confluence spaces
- search_confluence_using_cql: Search for existing pages using Confluence Query Language
- get_confluence_page: Get detailed content of specific pages

**Guidelines:**
- Focus on user-facing documentation, API docs, and architectural changes
- Consider both direct impacts (new features) and indirect impacts (changed behaviors)
- Prioritize high-impact changes that affect end users or developers
- Be specific about what sections of documentation need updating
- Consider creating new pages for significant new features
- Always verify existing documentation before recommending updates

Return a structured analysis with specific, actionable recommendations."""
)

@confluence_agent.tool
async def get_confluence_spaces(ctx: RunContext[AnalysisDependencies]) -> str:
    """Get all available Confluence spaces with their keys and names."""
    try:
        print("🔍 Getting Confluence spaces...")
        spaces = await ctx.deps.confluence_toolkit.get_confluence_spaces()
        simplified = [{"key": s.get("key"), "name": s.get("name")} for s in spaces]
        return json.dumps(simplified, indent=2)
    except Exception as e:
        raise ModelRetry(f"Error getting Confluence spaces: {e}")

@confluence_agent.tool
async def search_confluence_using_cql(ctx: RunContext[AnalysisDependencies], cql: str) -> str:
    """Search Confluence using a CQL query string."""
    try:
        print(f"🔍 Searching Confluence with CQL: {cql}")
        results = await ctx.deps.confluence_toolkit.search_confluence_using_cql(cql)
        if not results:
            return "No pages found for this CQL query."
        simplified = [{"id": r.get("id"), "title": r.get("title"), "space": r.get("space", {}).get("key")} for r in results]
        return json.dumps(simplified, indent=2)
    except Exception as e:
        raise ModelRetry(f"Error searching Confluence with CQL '{cql}': {e}")

@confluence_agent.tool
async def get_confluence_page(ctx: RunContext[AnalysisDependencies], page_id: str) -> str:
    """Get detailed content of a specific Confluence page by its ID."""
    try:
        print(f"📄 Getting Confluence page: {page_id}")
        page = await ctx.deps.confluence_toolkit.get_confluence_page(page_id)
        if not page or "error" in page:
            raise ModelRetry(f"Page with ID '{page_id}' not found or error retrieving it.")
        simplified = {
            "id": page.get("id"), 
            "title": page.get("title"), 
            "content_preview": page.get("body", {}).get("storage", {}).get("value", "")[:1000]
        }
        return json.dumps(simplified, indent=2)
    except Exception as e:
        raise ModelRetry(f"Error getting Confluence page '{page_id}': {e}")

# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class PRConfluenceAnalyzer:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.confluence_toolkit = ConfluenceToolkit(
            url=os.getenv('CONFLUENCE_URL'),
            username=os.getenv('CONFLUENCE_USERNAME'),
            api_token=os.getenv('CONFLUENCE_API_TOKEN')
        )
        self.pr_number = os.getenv('PR_NUMBER')
        self.repo_name = os.getenv('REPO_NAME')
        self.pr_title = os.getenv('PR_TITLE', '')
        self.pr_body = os.getenv('PR_BODY', '')
    
    async def get_pr_changes(self) -> List[Dict]:
        """Get PR file changes from GitHub API using httpx."""
        url = f"https://api.github.com/repos/{self.repo_name}/pulls/{self.pr_number}/files"
        headers = {
            'Authorization': f'token {os.getenv("GITHUB_TOKEN")}', 
            'Accept': 'application/vnd.github.v3+json'
        }
        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"❌ GitHub API error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"❌ Error fetching PR changes: {e}")
            return []
            
    def analyze_file_changes(self, files: List[Dict]) -> Dict:
        """Analyze the file changes to understand impact."""
        analysis = {
            'total_files': len(files), 
            'files_by_type': {}, 
            'significant_changes': []
        }
        
        for f in files:
            ext = f['filename'].split('.')[-1] if '.' in f['filename'] else 'no-ext'
            analysis['files_by_type'][ext] = analysis['files_by_type'].get(ext, 0) + 1
            
            if f.get('additions', 0) > 20 or f.get('deletions', 0) > 10:
                analysis['significant_changes'].append(f['filename'])
        
        return analysis
    
    async def run_analysis(self) -> ConfluenceAnalysisResult | None:
        """Run the main Confluence analysis using PydanticAI."""
        print("🔍 Getting PR file changes...")
        files = await self.get_pr_changes()
        if not files:
            print("❌ Could not retrieve PR file changes. Aborting.")
            return None
        
        file_analysis = self.analyze_file_changes(files)
        
        deps = AnalysisDependencies(
            client=self.client,
            confluence_toolkit=self.confluence_toolkit,
            pr_number=self.pr_number,
            repo_name=self.repo_name,
            pr_title=self.pr_title,
            pr_body=self.pr_body,
            file_analysis=file_analysis
        )
        
        analysis_prompt = f"""
        🚀 **CONFLUENCE DOCUMENTATION IMPACT ANALYSIS**
        **PR CONTEXT:** PR #{deps.pr_number} in {deps.repo_name}: "{deps.pr_title}"
        **CODE CHANGES:** {json.dumps(deps.file_analysis, indent=2)}
        **INSTRUCTIONS:** Analyze the PR context and code changes to identify necessary Confluence documentation updates. Use the available tools to explore Confluence, find relevant pages, and then formulate a detailed plan.
        """
        
        try:
            print("🤖 Starting PydanticAI analysis...")
            result = await confluence_agent.run(analysis_prompt, deps=deps)
            return result.data
        except Exception as e:
            print(f"❌ Error during PydanticAI analysis: {e}")
            return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    print("🚀 Starting PR Confluence Analysis with PydanticAI")
    print("=" * 60)
    
    required_vars = [
        'OPENAI_API_KEY', 
        'CONFLUENCE_URL', 
        'CONFLUENCE_USERNAME', 
        'CONFLUENCE_API_TOKEN', 
        'GITHUB_TOKEN', 
        'PR_NUMBER', 
        'REPO_NAME'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        return

    async with httpx.AsyncClient() as client:
        analyzer = PRConfluenceAnalyzer(client)
        print(f"📋 Analyzing PR #{analyzer.pr_number}: {analyzer.pr_title}")
        
        result = await analyzer.run_analysis()
        
        print("\n" + "="*60)
        print("📊 CONFLUENCE ANALYSIS RESULTS")
        print("="*60)
        
        if result:
            debug(result)  # Use devtools for rich output
            # Save structured results to a file
            with open('confluence_actions.json', 'w') as f:
                json.dump(result.model_dump(), f, indent=2)
            print("\n✅ Analysis successful. Structured results saved to confluence_actions.json")
        else:
            print("❌ Analysis failed.")
        
        print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())
