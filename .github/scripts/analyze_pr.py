from __future__ import annotations as _annotations

import os
import json
import asyncio
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Literal

import httpx
from devtools import debug
from atlassian import Confluence
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# ============================================================================
# ENHANCED CONFLUENCE TOOLKIT
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
        """Get a specific Confluence page with detailed content"""
        page = await asyncio.to_thread(
            self.confluence.get_page_by_id,
            page_id, 
            expand='body.storage,space,version,ancestors'
        )
        return page
    
    async def get_pages_in_confluence_space(self, space_key: str) -> List[Dict]:
        """Get all pages in a Confluence space"""
        pages = await asyncio.to_thread(
            self.confluence.get_all_pages_from_space,
            space=space_key, 
            start=0, 
            limit=50
        )
        return pages

# ============================================================================
# ENHANCED MODELS FOR BETTER ANALYSIS
# ============================================================================

@dataclass
class FileChangeAnalysis:
    """Detailed analysis of a single file change"""
    filename: str
    change_type: Literal["added", "modified", "deleted", "renamed"]
    file_category: str  # api, config, frontend, backend, test, docs, etc.
    impact_level: Literal["high", "medium", "low"]
    additions: int
    deletions: int
    is_breaking_change: bool
    affects_api: bool
    affects_user_interface: bool
    requires_migration: bool
    summary: str

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
    detailed_changes: List[FileChangeAnalysis]

class ContentSection(BaseModel):
    """A specific section of content that needs updating"""
    section_name: str = Field(description="Name of the section to update (e.g., 'API Reference', 'Installation Guide')")
    current_content_summary: Optional[str] = Field(description="Brief summary of current content in this section")
    required_changes: str = Field(description="Detailed description of what needs to be changed")
    example_content: str = Field(description="Example of new or updated content to add")
    change_type: Literal["add", "update", "remove", "restructure"] = Field(description="Type of change needed")

class ConfluenceAction(BaseModel):
    """A comprehensive Confluence documentation action with detailed guidance"""
    action: Literal["update_page", "create_page", "review_page", "archive_page"] = Field(description="Type of action needed")
    page_id: Optional[str] = Field(description="Confluence page ID if updating existing page")
    space_key: str = Field(description="Confluence space key")
    page_title: str = Field(description="Title of the page to update or create")
    reason: str = Field(description="Why this documentation needs updating")
    priority: Literal["critical", "high", "medium", "low"] = Field(description="Priority level based on user impact")
    estimated_time: str = Field(description="Estimated time to complete (e.g., '30 minutes', '2 hours')")
    
    # Enhanced content guidance
    content_sections: List[ContentSection] = Field(description="Specific sections that need updating with examples")
    affected_audiences: List[str] = Field(description="Who will be affected (developers, end-users, admins, etc.)")
    related_pages: List[str] = Field(description="Other pages that might need updates as a result")
    
    # Change context
    change_category: str = Field(description="Category of change (API, Feature, Bugfix, Configuration, etc.)")
    breaking_changes: bool = Field(description="Whether this involves breaking changes")
    migration_required: bool = Field(description="Whether users need to migrate/update their implementations")
    
    # Implementation guidance
    before_after_examples: Optional[str] = Field(description="Before/after examples showing the changes")
    implementation_notes: Optional[str] = Field(description="Technical notes for implementing the documentation updates")

class ConfluenceAnalysisResult(BaseModel):
    """Comprehensive analysis result with detailed recommendations"""
    confluence_actions: List[ConfluenceAction] = Field(description="List of detailed documentation actions needed")
    summary: str = Field(description="Executive summary of documentation impact and rationale")
    total_actions: int = Field(description="Total number of actions identified")
    estimated_total_effort: str = Field(description="Total estimated effort (e.g., '4-6 hours')")
    spaces_affected: List[str] = Field(description="List of Confluence spaces that will be affected")
    
    # Enhanced analysis
    change_categories: List[str] = Field(description="Categories of changes detected")
    critical_updates: List[str] = Field(description="Most critical updates that must be done immediately")
    user_impact_summary: str = Field(description="Summary of how these changes will impact different user types")
    rollout_recommendations: str = Field(description="Recommendations for rolling out documentation updates")

# ============================================================================
# ENHANCED FILE ANALYSIS
# ============================================================================

class EnhancedFileAnalyzer:
    """Advanced file change analyzer that categorizes and assesses impact"""
    
    @staticmethod
    def categorize_file(filename: str) -> str:
        """Categorize file based on path and extension"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['api', 'endpoint', 'route', 'controller']):
            return "api"
        elif any(keyword in filename_lower for keyword in ['config', 'setting', 'env', '.yaml', '.yml', '.json', '.toml']):
            return "configuration"
        elif any(keyword in filename_lower for keyword in ['frontend', 'ui', 'component', 'react', 'vue', 'angular']):
            return "frontend"
        elif any(keyword in filename_lower for keyword in ['backend', 'server', 'service', 'model', 'database']):
            return "backend"
        elif any(keyword in filename_lower for keyword in ['test', 'spec', '__test__']):
            return "test"
        elif any(keyword in filename_lower for keyword in ['doc', 'readme', 'md', 'rst']):
            return "documentation"
        elif any(keyword in filename_lower for keyword in ['migration', 'schema', 'sql']):
            return "database"
        elif any(keyword in filename_lower for keyword in ['deployment', 'docker', 'k8s', 'kubernetes', 'terraform']):
            return "infrastructure"
        else:
            return "other"
    
    @staticmethod
    def assess_breaking_change(filename: str, additions: int, deletions: int) -> bool:
        """Assess if this change is likely to be breaking"""
        if any(keyword in filename.lower() for keyword in ['api', 'interface', 'contract', 'schema']):
            return deletions > 10 or additions > 50
        return False
    
    @staticmethod
    def assess_impact_level(category: str, additions: int, deletions: int, is_breaking: bool) -> str:
        """Assess the impact level of the change"""
        if is_breaking or category == "api":
            return "high"
        elif category in ["configuration", "database", "infrastructure"] or additions > 100:
            return "medium"
        else:
            return "low"
    
    @classmethod
    def analyze_file_change(cls, file_data: Dict) -> FileChangeAnalysis:
        """Perform detailed analysis of a single file change"""
        filename = file_data.get('filename', '')
        status = file_data.get('status', 'modified')
        additions = file_data.get('additions', 0)
        deletions = file_data.get('deletions', 0)
        
        category = cls.categorize_file(filename)
        is_breaking = cls.assess_breaking_change(filename, additions, deletions)
        impact_level = cls.assess_impact_level(category, additions, deletions, is_breaking)
        
        # Generate summary
        summary = f"{status.title()} {category} file with {additions} additions and {deletions} deletions"
        if is_breaking:
            summary += " (potentially breaking)"
            
        return FileChangeAnalysis(
            filename=filename,
            change_type=status,
            file_category=category,
            impact_level=impact_level,
            additions=additions,
            deletions=deletions,
            is_breaking_change=is_breaking,
            affects_api=category == "api" or "api" in filename.lower(),
            affects_user_interface=category == "frontend",
            requires_migration=category in ["database", "api"] and is_breaking,
            summary=summary
        )

# ============================================================================
# ENHANCED PYDANTICAI AGENT
# ============================================================================

provider = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))
model = OpenAIModel('gpt-4.1', provider=provider)

confluence_agent = Agent[AnalysisDependencies, ConfluenceAnalysisResult](
    model,
    deps_type=AnalysisDependencies,
    output_type=ConfluenceAnalysisResult,
    retries=3,
    system_prompt="""You are an expert technical documentation strategist and content architect. Your mission is to analyze pull requests and provide comprehensive, actionable Confluence documentation update plans.

## ANALYSIS FRAMEWORK

### 1. CHANGE IMPACT ASSESSMENT
For each change, evaluate:
- **User Impact**: How will this affect different user types (developers, end-users, admins)?
- **Documentation Scope**: What documentation areas are affected?
- **Priority Level**: Critical (affects production), High (affects workflows), Medium (improves experience), Low (nice-to-have)
- **Effort Estimation**: Realistic time estimates for documentation work

### 2. CONTENT STRATEGY
Provide specific guidance on:
- **What to Write**: Exact sections and content types needed
- **How to Structure**: Organization and flow recommendations  
- **Examples to Include**: Code samples, screenshots, step-by-step guides
- **Audience Targeting**: Tailor content for specific user personas

### 3. CHANGE CATEGORIES TO DETECT
- **API Changes**: New endpoints, modified parameters, deprecated methods
- **Feature Additions**: New functionality, UI components, workflows
- **Configuration Changes**: New settings, environment variables, deployment options
- **Breaking Changes**: Backwards incompatible modifications requiring user action
- **Security Updates**: Authentication, authorization, security best practices
- **Performance Improvements**: Optimization tips, new performance features
- **Bug Fixes**: Corrections to documented behavior, troubleshooting updates

### 4. DOCUMENTATION TYPES TO CONSIDER
- **API Documentation**: OpenAPI specs, endpoint descriptions, parameter details
- **User Guides**: Step-by-step tutorials, feature explanations
- **Developer Documentation**: SDK guides, integration examples, technical references
- **Configuration Guides**: Setup instructions, environment configuration
- **Migration Guides**: Upgrade instructions, breaking change mitigation
- **Troubleshooting**: Known issues, error resolution, FAQ updates
- **Architecture Documentation**: System design, component relationships
- **Release Notes**: Change summaries, version histories

### 5. SPECIFIC CONTENT EXAMPLES TO PROVIDE
For each documentation action, include:
- **Before/After Code Examples**: Show old vs new implementations
- **Step-by-Step Instructions**: Numbered procedures for complex changes
- **Configuration Samples**: Complete config file examples
- **Migration Scripts**: Code snippets for upgrading implementations
- **Warning Callouts**: Important notes about breaking changes
- **Best Practice Recommendations**: Optimal ways to use new features

### 6. SEARCH STRATEGY
Use tools strategically:
1. **get_confluence_spaces**: Map available documentation spaces
2. **search_confluence_using_cql**: Find existing relevant pages using targeted queries like:
   - `space = "DEV" AND title ~ "API"` for API documentation
   - `space = "USER" AND text ~ "configuration"` for config docs
   - `type = "page" AND label = "migration"` for migration guides
3. **get_confluence_page**: Analyze existing content to understand current structure and identify gaps

### 7. QUALITY STANDARDS
Ensure recommendations include:
- **Accuracy**: Technically correct information
- **Completeness**: Cover all aspects of the change
- **Clarity**: Easy to understand for the target audience
- **Actionability**: Specific steps users can follow
- **Maintainability**: Updates that will stay current over time

### 8. OUTPUT REQUIREMENTS
Structure your analysis to include:
- **Executive Summary**: High-level impact and effort overview
- **Prioritized Actions**: Critical updates first, with time estimates
- **Detailed Content Plans**: Specific sections, examples, and guidance
- **Implementation Sequence**: Logical order for completing updates
- **Success Metrics**: How to measure documentation effectiveness

Remember: Great documentation prevents support tickets, reduces onboarding time, and enables users to succeed independently. Focus on creating content that truly serves the user's needs and business objectives."""
)

@confluence_agent.tool
async def get_confluence_spaces(ctx: RunContext[AnalysisDependencies]) -> str:
    """Get all available Confluence spaces with their keys, names, and descriptions."""
    try:
        print("🔍 Getting Confluence spaces...")
        spaces = await ctx.deps.confluence_toolkit.get_confluence_spaces()
        
        enhanced_spaces = []
        for s in spaces:
            space_info = {
                "key": s.get("key"),
                "name": s.get("name"),
                "description": s.get("description", {}).get("plain", {}).get("value", "No description"),
                "type": s.get("type"),
                "status": s.get("status")
            }
            enhanced_spaces.append(space_info)
        
        return json.dumps(enhanced_spaces, indent=2)
    except Exception as e:
        raise ModelRetry(f"Error getting Confluence spaces: {e}")

@confluence_agent.tool
async def search_confluence_using_cql(ctx: RunContext[AnalysisDependencies], cql: str) -> str:
    """Search Confluence using CQL with enhanced result details."""
    try:
        print(f"🔍 Searching Confluence with CQL: {cql}")
        results = await ctx.deps.confluence_toolkit.search_confluence_using_cql(cql)
        
        if not results:
            return "No pages found for this CQL query."
        
        enhanced_results = []
        for r in results:
            result_info = {
                "id": r.get("id"),
                "title": r.get("title"),
                "space": r.get("space", {}).get("key"),
                "space_name": r.get("space", {}).get("name"),
                "url": r.get("_links", {}).get("webui"),
                "excerpt": r.get("excerpt", ""),
                "last_modified": r.get("lastModified"),
                "content_type": r.get("content", {}).get("type", "page")
            }
            enhanced_results.append(result_info)
        
        return json.dumps(enhanced_results, indent=2)
    except Exception as e:
        raise ModelRetry(f"Error searching Confluence with CQL '{cql}': {e}")

@confluence_agent.tool
async def get_confluence_page(ctx: RunContext[AnalysisDependencies], page_id: str) -> str:
    """Get detailed content and structure of a specific Confluence page."""
    try:
        print(f"📄 Getting Confluence page: {page_id}")
        page = await ctx.deps.confluence_toolkit.get_confluence_page(page_id)
        
        if not page or "error" in page:
            raise ModelRetry(f"Page with ID '{page_id}' not found or error retrieving it.")
        
        # Extract structured content information
        content = page.get("body", {}).get("storage", {}).get("value", "")
        
        # Parse content structure (headings, sections)
        headings = re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', content, re.IGNORECASE)
        
        page_info = {
            "id": page.get("id"),
            "title": page.get("title"),
            "space_key": page.get("space", {}).get("key"),
            "space_name": page.get("space", {}).get("name"),
            "version": page.get("version", {}).get("number"),
            "last_modified": page.get("version", {}).get("when"),
            "url": page.get("_links", {}).get("webui"),
            "content_length": len(content),
            "headings": headings[:10],  # First 10 headings to understand structure
            "content_preview": content[:2000] if content else "No content",
            "has_code_blocks": "<code>" in content or "```" in content,
            "has_tables": "<table>" in content,
            "has_images": "<img" in content or "confluence-embedded-image" in content
        }
        
        return json.dumps(page_info, indent=2)
    except Exception as e:
        raise ModelRetry(f"Error getting Confluence page '{page_id}': {e}")

@confluence_agent.tool
async def search_related_documentation(ctx: RunContext[AnalysisDependencies], search_terms: List[str]) -> str:
    """Search for documentation related to the PR changes using intelligent queries."""
    try:
        print(f"🔍 Searching for related documentation: {search_terms}")
        
        all_results = []
        
        # Search across different content types and spaces
        search_strategies = [
            f"text ~ \"{term}\"" for term in search_terms[:3]  # Limit to top 3 terms
        ]
        
        # Add category-specific searches based on file changes
        if any(change.affects_api for change in ctx.deps.detailed_changes):
            search_strategies.append("(title ~ \"API\" OR text ~ \"endpoint\" OR text ~ \"REST\")")
        
        if any(change.file_category == "configuration" for change in ctx.deps.detailed_changes):
            search_strategies.append("(title ~ \"config\" OR text ~ \"configuration\" OR text ~ \"setup\")")
        
        for strategy in search_strategies:
            try:
                results = await ctx.deps.confluence_toolkit.search_confluence_using_cql(strategy)
                all_results.extend(results[:5])  # Top 5 results per strategy
            except Exception:
                continue  # Skip failed searches
        
        # Deduplicate by page ID
        unique_results = {r.get("id"): r for r in all_results}.values()
        
        simplified_results = [
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "space": r.get("space", {}).get("key"),
                "relevance_score": "high" if any(term.lower() in r.get("title", "").lower() for term in search_terms) else "medium"
            }
            for r in list(unique_results)[:10]  # Top 10 unique results
        ]
        
        return json.dumps(simplified_results, indent=2)
    except Exception as e:
        raise ModelRetry(f"Error searching related documentation: {e}")

# ============================================================================
# ENHANCED ANALYZER CLASS
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
        self.file_analyzer = EnhancedFileAnalyzer()
    
    async def get_pr_changes(self) -> List[Dict]:
        """Get PR file changes from GitHub API."""
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
    
    def analyze_file_changes(self, files: List[Dict]) -> tuple[Dict, List[FileChangeAnalysis]]:
        """Enhanced file analysis with detailed categorization."""
        detailed_changes = [
            self.file_analyzer.analyze_file_change(f) for f in files
        ]
        
        # Generate comprehensive summary
        analysis = {
            'total_files': len(files),
            'files_by_category': {},
            'files_by_impact': {'high': 0, 'medium': 0, 'low': 0},
            'breaking_changes': [],
            'api_changes': [],
            'config_changes': [],
            'significant_changes': [],
            'change_summary': ""
        }
        
        for change in detailed_changes:
            # Category breakdown
            category = change.file_category
            analysis['files_by_category'][category] = analysis['files_by_category'].get(category, 0) + 1
            
            # Impact breakdown
            analysis['files_by_impact'][change.impact_level] += 1
            
            # Special change tracking
            if change.is_breaking_change:
                analysis['breaking_changes'].append(change.filename)
            if change.affects_api:
                analysis['api_changes'].append(change.filename)
            if change.file_category == 'configuration':
                analysis['config_changes'].append(change.filename)
            if change.additions > 20 or change.deletions > 10:
                analysis['significant_changes'].append(change.filename)
        
        # Generate change summary
        high_impact = analysis['files_by_impact']['high']
        categories = list(analysis['files_by_category'].keys())
        
        summary_parts = [f"{high_impact} high-impact changes" if high_impact else "No high-impact changes"]
        if categories:
            summary_parts.append(f"affecting {', '.join(categories[:3])}")
        if analysis['breaking_changes']:
            summary_parts.append(f"with {len(analysis['breaking_changes'])} potential breaking changes")
        
        analysis['change_summary'] = ", ".join(summary_parts)
        
        return analysis, detailed_changes
    
    async def run_analysis(self) -> ConfluenceAnalysisResult | None:
        """Run comprehensive Confluence analysis."""
        print("🔍 Getting PR file changes...")
        files = await self.get_pr_changes()
        if not files:
            print("❌ Could not retrieve PR file changes. Aborting.")
            return None
        
        print("📊 Analyzing file changes...")
        file_analysis, detailed_changes = self.analyze_file_changes(files)
        
        deps = AnalysisDependencies(
            client=self.client,
            confluence_toolkit=self.confluence_toolkit,
            pr_number=self.pr_number,
            repo_name=self.repo_name,
            pr_title=self.pr_title,
            pr_body=self.pr_body,
            file_analysis=file_analysis,
            detailed_changes=detailed_changes
        )
        
        # Enhanced analysis prompt with more context
        analysis_prompt = f"""
🚀 **COMPREHENSIVE CONFLUENCE DOCUMENTATION ANALYSIS**

## PR CONTEXT
- **Repository**: {deps.repo_name}
- **PR #{deps.pr_number}**: {deps.pr_title}
- **Description**: {deps.pr_body[:500]}...

## CHANGE ANALYSIS
{json.dumps(deps.file_analysis, indent=2)}

## DETAILED FILE CHANGES
{json.dumps([change.__dict__ for change in deps.detailed_changes], indent=2, default=str)}

## ANALYSIS OBJECTIVES
1. **Identify Documentation Gaps**: What existing documentation needs updating?
2. **Determine New Content Needs**: What new documentation should be created?
3. **Assess User Impact**: How will different user types be affected?
4. **Prioritize Actions**: What needs immediate attention vs. future updates?
5. **Provide Implementation Guidance**: Specific examples and content recommendations

## INSTRUCTIONS
Use the available tools to:
1. Explore Confluence spaces to understand documentation structure
2. Search for existing documentation related to the changed components
3. Analyze current content to identify specific sections needing updates
4. Formulate detailed, actionable recommendations with examples

Focus on creating documentation that prevents confusion, reduces support burden, and enables users to successfully implement or adapt to these changes.
        """
        
        try:
            print("🤖 Starting comprehensive PydanticAI analysis...")
            result = await confluence_agent.run(analysis_prompt, deps=deps)
            return result.data
        except Exception as e:
            print(f"❌ Error during PydanticAI analysis: {e}")
            return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    print("🚀 Starting Enhanced PR Confluence Analysis")
    print("=" * 70)
    
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        analyzer = PRConfluenceAnalyzer(client)
        print(f"📋 Analyzing PR #{analyzer.pr_number}: {analyzer.pr_title}")
        
        result = await analyzer.run_analysis()
        
        print("\n" + "="*70)
        print("📊 ENHANCED CONFLUENCE ANALYSIS RESULTS")
        print("="*70)
        
        if result:
            # Pretty print with devtools
            debug(result)
            
            # Save detailed results
            output_data = result.model_dump()
            with open('confluence_actions.json', 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Generate summary report
            print(f"\n📋 EXECUTIVE SUMMARY")
            print(f"Total Actions: {result.total_actions}")
            print(f"Estimated Effort: {result.estimated_total_effort}")
            print(f"Spaces Affected: {', '.join(result.spaces_affected)}")
            print(f"Change Categories: {', '.join(result.change_categories)}")
            
            if result.critical_updates:
                print(f"\n🚨 CRITICAL UPDATES:")
                for update in result.critical_updates:
                    print(f"  • {update}")
            
            print("\n✅ Enhanced analysis complete! Detailed results saved to confluence_actions.json")
        else:
            print("❌ Analysis failed.")
        
        print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())
