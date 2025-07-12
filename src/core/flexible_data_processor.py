#!/usr/bin/env python3
"""
RELOAD FLEXIBLE DATA PROCESSOR FOR AGENT LOBBI
===========================================
Makes the lobby more tolerant of different data formats and handles
standardization internally rather than requiring agents to format data precisely.

Key Features:
- Field name normalization (handles variations and synonyms)
- Intelligent type conversion and auto-correction
- Graceful handling of encoding issues
- Smart defaults for missing fields
- Flexible capability detection
- Multiple input format support
"""

import json
import re
import html
import urllib.parse
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of flexible data processing"""
    success: bool
    standardized_data: Dict[str, Any]
    warnings: List[str]
    corrections: List[str]  # What was auto-corrected
    original_issues: List[str]  # Issues that were fixed
    
class FlexibleDataProcessor:
    """
    Handles data format flexibility and normalization for Agent Lobbi.
    Shifts the burden of data formatting from agents to the lobby itself.
    """
    
    # Field name mapping - handles various ways agents might name fields
    FIELD_ALIASES = {
        'task_title': ['title', 'task_name', 'name', 'task', 'subject', 'task_title'],
        'task_description': ['description', 'desc', 'details', 'content', 'body', 'task_description', 'task_details'],
        'required_capabilities': ['capabilities', 'skills', 'requirements', 'needs', 'required_skills', 'caps', 'required_capabilities'],
        'requester_id': ['requester', 'user_id', 'client_id', 'requester_id', 'from', 'sender', 'user'],
        'task_intent': ['intent', 'goal', 'purpose', 'objective', 'task_intent', 'collaboration_goal'],
        'max_agents': ['max_agents', 'agent_count', 'num_agents', 'agent_limit', 'max_workers'],
        'task_data': ['data', 'payload', 'extra_data', 'task_data', 'additional_data', 'context']
    }
    
    # Default values for missing required fields
    SMART_DEFAULTS = {
        'task_title': 'Untitled Task',
        'task_description': 'No description provided',
        'required_capabilities': ['general'],
        'requester_id': 'anonymous_user',
        'task_intent': '',
        'max_agents': 1,
        'task_data': {}
    }
    
    # Common capability aliases
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
        self.processing_stats = {
            'total_processed': 0,
            'corrections_made': 0,
            'defaults_applied': 0,
            'format_conversions': 0
        }
    
    def process_task_delegation(self, raw_data: Any) -> ProcessingResult:
        """
        Main entry point - processes raw task delegation data into standardized format.
        Handles various input formats and makes intelligent corrections.
        """
        self.processing_stats['total_processed'] += 1
        warnings = []
        corrections = []
        original_issues = []
        
        # Step 1: Handle different input formats
        try:
            normalized_input = self._normalize_input_format(raw_data)
            if normalized_input != raw_data:
                corrections.append("Converted input format to dictionary")
                self.processing_stats['format_conversions'] += 1
        except Exception as e:
            return ProcessingResult(
                success=False,
                standardized_data={},
                warnings=[f"Failed to parse input: {str(e)}"],
                corrections=[],
                original_issues=[f"Invalid input format: {type(raw_data)}"]
            )
        
        # Step 2: Normalize field names
        field_mapped_data = self._normalize_field_names(normalized_input)
        if field_mapped_data != normalized_input:
            corrections.append("Mapped field names to standard format")
        
        # Step 3: Handle encoding and text issues
        cleaned_data = self._handle_encoding_issues(field_mapped_data)
        if cleaned_data != field_mapped_data:
            corrections.append("Fixed text encoding issues")
        
        # Step 4: Validate and convert data types
        type_corrected_data, type_warnings = self._validate_and_convert_types(cleaned_data)
        warnings.extend(type_warnings)
        if type_corrected_data != cleaned_data:
            corrections.append("Corrected data types")
        
        # Step 5: Apply smart defaults for missing fields
        standardized_data, default_warnings = self._apply_smart_defaults(type_corrected_data)
        warnings.extend(default_warnings)
        if len(default_warnings) > 0:
            self.processing_stats['defaults_applied'] += 1
        
        # Step 6: Normalize capabilities
        standardized_data = self._normalize_capabilities(standardized_data)
        
        # Step 7: Final validation and cleanup
        final_data, final_warnings = self._final_validation_and_cleanup(standardized_data)
        warnings.extend(final_warnings)
        
        if len(corrections) > 0:
            self.processing_stats['corrections_made'] += 1
        
        return ProcessingResult(
            success=True,
            standardized_data=final_data,
            warnings=warnings,
            corrections=corrections,
            original_issues=original_issues
        )
    
    def _normalize_input_format(self, raw_data: Any) -> Dict[str, Any]:
        """Handle different input formats - strings, lists, objects, etc."""
        
        # Already a dictionary - ideal case
        if isinstance(raw_data, dict):
            return raw_data
        
        # JSON string
        if isinstance(raw_data, str):
            try:
                parsed = json.loads(raw_data)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Maybe it's a simple string description
            return {
                'task_description': raw_data,
                'task_title': raw_data[:50] + "..." if len(raw_data) > 50 else raw_data
            }
        
        # List format - might be [title, description, capabilities]
        if isinstance(raw_data, list) and len(raw_data) >= 1:
            result = {}
            if len(raw_data) >= 1:
                result['task_title'] = str(raw_data[0])
            if len(raw_data) >= 2:
                result['task_description'] = str(raw_data[1])
            if len(raw_data) >= 3:
                if isinstance(raw_data[2], list):
                    result['required_capabilities'] = raw_data[2]
                else:
                    result['required_capabilities'] = [str(raw_data[2])]
            return result
        
        # Fallback - convert to string and treat as description
        return {
            'task_description': str(raw_data),
            'task_title': f"Task from {type(raw_data).__name__}"
        }
    
    def _normalize_field_names(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map various field name variations to standard names"""
        normalized = {}
        used_keys = set()
        
        # Map known aliases to standard field names
        for standard_field, aliases in self.FIELD_ALIASES.items():
            for alias in aliases:
                if alias in data and alias not in used_keys:
                    normalized[standard_field] = data[alias]
                    used_keys.add(alias)
                    break
        
        # Copy any unmapped fields that don't conflict
        for key, value in data.items():
            if key not in used_keys and key not in normalized:
                normalized[key] = value
        
        return normalized
    
    def _handle_encoding_issues(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common encoding and text issues"""
        cleaned = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Fix common encoding issues
                try:
                    # Handle UTF-8 encoding issues
                    if isinstance(value, bytes):
                        value = value.decode('utf-8', errors='replace')
                    
                    # Fix HTML entities
                    value = html.unescape(value)
                    
                    # Fix URL encoding
                    value = urllib.parse.unquote(value)
                    
                    # Remove null bytes and control characters
                    value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
                    
                    # Normalize whitespace
                    value = ' '.join(value.split())
                    
                    cleaned[key] = value
                    
                except Exception:
                    # If all else fails, convert to string safely
                    cleaned[key] = str(value)
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _validate_and_convert_types(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate and convert data types with intelligent correction"""
        corrected = {}
        warnings = []
        
        for key, value in data.items():
            if key == 'required_capabilities':
                # Ensure capabilities is a list
                if isinstance(value, str):
                    # Split comma-separated string
                    corrected[key] = [cap.strip() for cap in value.split(',') if cap.strip()]
                    warnings.append(f"Converted capabilities from string to list")
                elif isinstance(value, list):
                    corrected[key] = [str(item).strip() for item in value if str(item).strip()]
                else:
                    corrected[key] = [str(value)]
                    warnings.append(f"Converted capabilities to list format")
            
            elif key == 'max_agents':
                # Ensure max_agents is an integer
                try:
                    if isinstance(value, str):
                        corrected[key] = int(value)
                        warnings.append("Converted max_agents from string to integer")
                    elif isinstance(value, (int, float)):
                        corrected[key] = int(value)
                    else:
                        corrected[key] = 1
                        warnings.append("Set max_agents to default value (1)")
                except ValueError:
                    corrected[key] = 1
                    warnings.append("Invalid max_agents value, set to 1")
            
            elif key in ['task_title', 'task_description', 'requester_id', 'task_intent']:
                # Ensure text fields are strings
                corrected[key] = str(value) if value is not None else ""
            
            elif key == 'task_data':
                # Ensure task_data is a dictionary
                if isinstance(value, dict):
                    corrected[key] = value
                elif isinstance(value, str):
                    try:
                        corrected[key] = json.loads(value)
                        warnings.append("Parsed task_data from JSON string")
                    except json.JSONDecodeError:
                        corrected[key] = {'raw_data': value}
                        warnings.append("Wrapped task_data in dictionary")
                else:
                    corrected[key] = {'data': value}
                    warnings.append("Converted task_data to dictionary format")
            
            else:
                # Keep other fields as-is
                corrected[key] = value
        
        return corrected, warnings
    
    def _apply_smart_defaults(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply intelligent defaults for missing required fields"""
        completed = data.copy()
        warnings = []
        
        for field, default_value in self.SMART_DEFAULTS.items():
            if field not in completed or not completed[field]:
                completed[field] = default_value
                warnings.append(f"Applied default value for '{field}'")
        
        # Special logic for generating better defaults based on available data
        if completed['task_title'] == self.SMART_DEFAULTS['task_title']:
            # Try to generate title from description
            desc = completed.get('task_description', '')
            if desc and desc != self.SMART_DEFAULTS['task_description']:
                # Use first sentence or first 50 chars as title
                first_sentence = desc.split('.')[0]
                if len(first_sentence) <= 50:
                    completed['task_title'] = first_sentence
                else:
                    completed['task_title'] = desc[:47] + "..."
                warnings.append("Generated task title from description")
        
        return completed, warnings
    
    def _normalize_capabilities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize capability names using aliases"""
        if 'required_capabilities' not in data:
            return data
        
        capabilities = data['required_capabilities']
        normalized_caps = []
        
        for cap in capabilities:
            cap_lower = cap.lower().strip()
            
            # Check if it matches any standard capability
            found = False
            for standard_cap, aliases in self.CAPABILITY_ALIASES.items():
                if cap_lower == standard_cap or cap_lower in aliases:
                    normalized_caps.append(standard_cap)
                    found = True
                    break
            
            # If not found, keep original but clean it up
            if not found:
                # Remove special characters and normalize
                clean_cap = re.sub(r'[^a-zA-Z0-9_]', '_', cap).lower()
                normalized_caps.append(clean_cap)
        
        data['required_capabilities'] = normalized_caps if normalized_caps else ['general']
        return data
    
    def _final_validation_and_cleanup(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Final validation and cleanup"""
        cleaned = data.copy()
        warnings = []
        
        # Ensure required fields are not empty
        if not cleaned.get('task_title', '').strip():
            cleaned['task_title'] = "Task"
            warnings.append("Set empty task title to default")
        
        if not cleaned.get('task_description', '').strip():
            cleaned['task_description'] = f"Task: {cleaned['task_title']}"
            warnings.append("Generated description from title")
        
        if not cleaned.get('required_capabilities'):
            cleaned['required_capabilities'] = ['general']
            warnings.append("Set empty capabilities to 'general'")
        
        # Limit lengths to prevent issues
        if len(cleaned['task_title']) > 200:
            cleaned['task_title'] = cleaned['task_title'][:197] + "..."
            warnings.append("Truncated long task title")
        
        if len(cleaned['task_description']) > 2000:
            cleaned['task_description'] = cleaned['task_description'][:1997] + "..."
            warnings.append("Truncated long task description")
        
        # Ensure max_agents is reasonable
        if cleaned['max_agents'] < 1:
            cleaned['max_agents'] = 1
            warnings.append("Set max_agents minimum to 1")
        elif cleaned['max_agents'] > 10:
            cleaned['max_agents'] = 10
            warnings.append("Limited max_agents to maximum of 10")
        
        return cleaned, warnings
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about data processing"""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'corrections_made': 0,
            'defaults_applied': 0,
            'format_conversions': 0
        }

# Convenience function for quick processing
def process_flexible_data(raw_data: Any) -> Dict[str, Any]:
    """Quick processing function that returns standardized data"""
    processor = FlexibleDataProcessor()
    result = processor.process_task_delegation(raw_data)
    
    if result.success:
        return result.standardized_data
    else:
        # Return basic fallback data
        return {
            'task_title': 'Failed Task Processing',
            'task_description': f'Could not process input: {result.warnings}',
            'required_capabilities': ['general'],
            'requester_id': 'system',
            'task_intent': '',
            'max_agents': 1,
            'task_data': {'original_input': str(raw_data)}
        } 