#!/usr/bin/env python3
"""
DART CAPABILITY MATCHER FOR AGENT LOBBI
=====================================
Handles capability aliases and matching to ensure agents can be found
even when capability names don't match exactly.

This solves the issue where agents register with 'creative_writing' 
but tasks request 'writing' capabilities.
"""

from typing import List, Dict, Set
import re

class CapabilityMatcher:
    """
    Handles capability matching with aliases and normalization.
    Integrates with the flexible data processor's capability aliases.
    """
    
    # Same aliases as in flexible_data_processor.py for consistency
    CAPABILITY_ALIASES = {
        'math': ['mathematical_analysis', 'mathematics', 'calculation', 'arithmetic', 'calculus'],
        'science': ['scientific_analysis', 'research', 'analysis', 'investigation'],
        'writing': ['creative_writing', 'content_creation', 'documentation', 'text_generation'],
        'code': ['software_development', 'programming', 'coding', 'development'],
        'data': ['data_analysis', 'data_science', 'analytics', 'statistics'],
        'finance': ['financial_analysis', 'economics', 'accounting', 'investment'],
        'general': ['any', 'generic', 'universal', 'default']
    }
    
    def __init__(self):
        # Build reverse mapping for fast lookups
        self._build_reverse_mapping()
    
    def _build_reverse_mapping(self):
        """Build reverse mapping from aliases to standard capabilities"""
        self.alias_to_standard = {}
        
        for standard_cap, aliases in self.CAPABILITY_ALIASES.items():
            # Standard capability maps to itself
            self.alias_to_standard[standard_cap] = standard_cap
            
            # Each alias maps to the standard capability
            for alias in aliases:
                self.alias_to_standard[alias] = standard_cap
    
    def normalize_capability(self, capability: str) -> str:
        """Normalize a capability name to its standard form"""
        # Type safety: Handle non-string capabilities gracefully
        if not isinstance(capability, str):
            if isinstance(capability, dict):
                # If it's a dict, try to extract a string value
                capability = capability.get('name', str(capability))
            else:
                capability = str(capability)
        
        capability_clean = capability.lower().strip()
        
        # Direct mapping
        if capability_clean in self.alias_to_standard:
            return self.alias_to_standard[capability_clean]
        
        # Fuzzy matching for partial matches
        for standard_cap, aliases in self.CAPABILITY_ALIASES.items():
            if capability_clean == standard_cap:
                return standard_cap
            
            # Check if capability contains any alias
            for alias in aliases:
                if alias in capability_clean or capability_clean in alias:
                    return standard_cap
        
        # If no match found, return normalized version of original
        return re.sub(r'[^a-zA-Z0-9_]', '_', capability_clean)
    
    def normalize_capabilities(self, capabilities: List[str]) -> List[str]:
        """Normalize a list of capabilities"""
        # Type safety: Handle non-list inputs gracefully
        if not isinstance(capabilities, (list, tuple)):
            capabilities = [capabilities] if capabilities else []
        
        normalized = []
        for cap in capabilities:
            try:
                norm_cap = self.normalize_capability(cap)
                if norm_cap and norm_cap not in normalized:  # Avoid duplicates and empty strings
                    normalized.append(norm_cap)
            except Exception as e:
                # Log warning but continue processing other capabilities
                print(f"Warning: Could not normalize capability {cap}: {e}")
                continue
        return normalized
    
    def capabilities_match(self, requested_capability: str, agent_capabilities: List[str]) -> bool:
        """Check if a requested capability matches any of the agent's capabilities"""
        requested_norm = self.normalize_capability(requested_capability)
        agent_norm = self.normalize_capabilities(agent_capabilities)
        
        return requested_norm in agent_norm
    
    def find_matching_agents(self, requested_capability: str, agents_dict: Dict) -> List[str]:
        """
        Find agents that have a capability matching the requested one.
        
        Args:
            requested_capability: The capability being requested
            agents_dict: Dictionary of agents {agent_id: {capabilities: [...]}}
            
        Returns:
            List of agent IDs that match the capability
        """
        matching_agents = []
        
        for agent_id, agent_info in agents_dict.items():
            if not isinstance(agent_info, dict):
                continue
                
            agent_capabilities = agent_info.get('capabilities', [])
            if self.capabilities_match(requested_capability, agent_capabilities):
                matching_agents.append(agent_id)
        
        return matching_agents
    
    def get_all_variations(self, capability: str) -> List[str]:
        """Get all variations (aliases) of a capability"""
        normalized = self.normalize_capability(capability)
        
        for standard_cap, aliases in self.CAPABILITY_ALIASES.items():
            if normalized == standard_cap:
                return [standard_cap] + aliases
        
        return [capability]
    
    def get_capability_info(self, capability: str) -> Dict[str, any]:
        """Get detailed information about a capability and its aliases"""
        normalized = self.normalize_capability(capability)
        variations = self.get_all_variations(capability)
        
        return {
            'original': capability,
            'normalized': normalized,
            'variations': variations,
            'is_standard': capability.lower() in self.CAPABILITY_ALIASES
        }

# Global instance for easy import
capability_matcher = CapabilityMatcher()

# Convenience functions
def normalize_capability(capability: str) -> str:
    """Normalize a single capability"""
    return capability_matcher.normalize_capability(capability)

def normalize_capabilities(capabilities: List[str]) -> List[str]:
    """Normalize a list of capabilities"""
    return capability_matcher.normalize_capabilities(capabilities)

def capabilities_match(requested: str, agent_capabilities: List[str]) -> bool:
    """Check if capabilities match"""
    return capability_matcher.capabilities_match(requested, agent_capabilities)

def find_matching_agents(requested_capability: str, agents_dict: Dict) -> List[str]:
    """Find agents that match a capability"""
    return capability_matcher.find_matching_agents(requested_capability, agents_dict) 