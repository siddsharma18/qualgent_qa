import re
import difflib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class ElementMatch:
    """Represents a matched UI element with confidence score."""
    element_id: str
    confidence: float
    strategy: str
    element_data: Dict[str, Any]
    matched_text: str

class UIParser:
    """
    A robust UI parser that uses multiple strategies to find UI elements
    based on natural language subgoals.
    """
    
    def __init__(self, 
                 min_confidence: float = 0.3,
                 enable_fuzzy_matching: bool = True,
                 enable_semantic_matching: bool = True):
        """
        Initialize the UI parser.
        
        Args:
            min_confidence: Minimum confidence threshold for element matching
            enable_fuzzy_matching: Whether to use fuzzy string matching
            enable_semantic_matching: Whether to use semantic matching
        """
        self.min_confidence = min_confidence
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.enable_semantic_matching = enable_semantic_matching
        
        # Common UI element keywords and their variations
        self.element_keywords = {
            'button': ['button', 'btn', 'tap', 'click', 'press', 'select'],
            'toggle': ['toggle', 'switch', 'on/off', 'enable', 'disable'],
            'text': ['text', 'label', 'title', 'heading'],
            'input': ['input', 'field', 'textbox', 'edit'],
            'menu': ['menu', 'options', 'settings', 'preferences'],
            'wifi': ['wifi', 'wi-fi', 'wireless', 'network'],
            'bluetooth': ['bluetooth', 'bluetooth', 'bt'],
            'brightness': ['brightness', 'screen brightness', 'display brightness'],
            'volume': ['volume', 'sound', 'audio'],
            'battery': ['battery', 'power', 'charging'],
            'storage': ['storage', 'memory', 'space'],
            'security': ['security', 'lock', 'password', 'fingerprint'],
            'accessibility': ['accessibility', 'a11y', 'assistive'],
            'about': ['about', 'info', 'information', 'device info'],
            'back': ['back', 'return', 'previous', 'go back'],
            'home': ['home', 'main', 'start'],
            'search': ['search', 'find', 'lookup'],
            'add': ['add', 'create', 'new', 'plus'],
            'delete': ['delete', 'remove', 'trash', 'bin'],
            'save': ['save', 'store', 'keep'],
            'cancel': ['cancel', 'abort', 'stop'],
            'ok': ['ok', 'confirm', 'yes', 'accept'],
            'no': ['no', 'deny', 'reject', 'decline']
        }
        
        # Action keywords mapping
        self.action_keywords = {
            'turn_on': ['turn on', 'enable', 'activate', 'start', 'open'],
            'turn_off': ['turn off', 'disable', 'deactivate', 'stop', 'close'],
            'toggle': ['toggle', 'switch', 'change'],
            'tap': ['tap', 'click', 'press', 'select', 'choose'],
            'type': ['type', 'enter', 'input', 'write'],
            'scroll': ['scroll', 'swipe', 'move'],
            'back': ['go back', 'return', 'previous'],
            'home': ['go home', 'main screen', 'start']
        }
    
    def find_element_for_subgoal(self, ui_tree: List[Dict[str, Any]], subgoal: str) -> Optional[str]:
        """
        Find the best matching UI element for a given subgoal.
        
        Args:
            ui_tree: The UI tree from the environment
            subgoal: The subgoal string (e.g., "Turn Wi-Fi off")
            
        Returns:
            element_id: The ID of the best matching element, or None if not found
        """
        if not ui_tree:
            return None
        
        # Try multiple strategies to find the element
        strategies = [
            self._exact_text_match,
            self._fuzzy_text_match,
            self._semantic_match,
            self._keyword_match,
            self._action_based_match
        ]
        
        best_match = None
        strategies_tried = []
        
        for strategy in strategies:
            try:
                match = strategy(ui_tree, subgoal)
                if match and match.confidence > self.min_confidence:
                    if best_match is None or match.confidence > best_match.confidence:
                        best_match = match
                strategies_tried.append(strategy.__name__)
            except Exception as e:
                logging.warning(f"Strategy {strategy.__name__} failed: {e}")
                strategies_tried.append(f"{strategy.__name__}(failed)")
        
        if best_match:
            logging.info(f"Found element {best_match.element_id} with confidence {best_match.confidence} using {best_match.strategy}")
            return best_match.element_id
        
        logging.warning(f"No element found for subgoal '{subgoal}'. Tried strategies: {strategies_tried}")
        return None
    
    def _exact_text_match(self, ui_tree: List[Dict[str, Any]], subgoal: str) -> Optional[ElementMatch]:
        """Find elements with exact text matches."""
        subgoal_lower = subgoal.lower()
        
        for element in ui_tree:
            text = self._extract_text(element)
            if text and text.lower() == subgoal_lower:
                return ElementMatch(
                    element_id=element.get('id', ''),
                    confidence=1.0,
                    strategy='exact_text_match',
                    element_data=element,
                    matched_text=text
                )
        
        return None
    
    def _fuzzy_text_match(self, ui_tree: List[Dict[str, Any]], subgoal: str) -> Optional[ElementMatch]:
        """Find elements using fuzzy string matching."""
        if not self.enable_fuzzy_matching:
            return None
        
        subgoal_lower = subgoal.lower()
        best_match = None
        best_ratio = 0
        
        for element in ui_tree:
            text = self._extract_text(element)
            if text:
                ratio = difflib.SequenceMatcher(None, text.lower(), subgoal_lower).ratio()
                if ratio > best_ratio and ratio > 0.6:  # Minimum threshold for fuzzy matching
                    best_ratio = ratio
                    best_match = ElementMatch(
                        element_id=element.get('id', ''),
                        confidence=ratio,
                        strategy='fuzzy_text_match',
                        element_data=element,
                        matched_text=text
                    )
        
        return best_match
    
    def _semantic_match(self, ui_tree: List[Dict[str, Any]], subgoal: str) -> Optional[ElementMatch]:
        """Find elements using semantic matching based on keywords."""
        if not self.enable_semantic_matching:
            return None
        
        subgoal_lower = subgoal.lower()
        best_match = None
        best_score = 0
        
        # Extract keywords from subgoal
        subgoal_keywords = self._extract_keywords(subgoal_lower)
        
        for element in ui_tree:
            text = self._extract_text(element)
            if text:
                element_keywords = self._extract_keywords(text.lower())
                
                # Calculate semantic similarity
                score = self._calculate_semantic_similarity(subgoal_keywords, element_keywords)
                
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = ElementMatch(
                        element_id=element.get('id', ''),
                        confidence=score,
                        strategy='semantic_match',
                        element_data=element,
                        matched_text=text
                    )
        
        return best_match
    
    def _keyword_match(self, ui_tree: List[Dict[str, Any]], subgoal: str) -> Optional[ElementMatch]:
        """Find elements by matching specific keywords."""
        subgoal_lower = subgoal.lower()
        best_match = None
        best_score = 0
        
        for element in ui_tree:
            text = self._extract_text(element)
            if text:
                text_lower = text.lower()
                score = 0
                
                # Check for keyword matches
                for category, keywords in self.element_keywords.items():
                    for keyword in keywords:
                        if keyword in text_lower and keyword in subgoal_lower:
                            score += 0.5
                        elif keyword in text_lower or keyword in subgoal_lower:
                            score += 0.2
                
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = ElementMatch(
                        element_id=element.get('id', ''),
                        confidence=min(score, 1.0),
                        strategy='keyword_match',
                        element_data=element,
                        matched_text=text
                    )
        
        return best_match
    
    def _action_based_match(self, ui_tree: List[Dict[str, Any]], subgoal: str) -> Optional[ElementMatch]:
        """Find elements based on the action type in the subgoal."""
        subgoal_lower = subgoal.lower()
        
        # Determine the action type
        action_type = None
        for action, keywords in self.action_keywords.items():
            if any(keyword in subgoal_lower for keyword in keywords):
                action_type = action
                break
        
        if not action_type:
            return None
        
        # Look for elements that match the action context
        for element in ui_tree:
            text = self._extract_text(element)
            if text:
                # Check if element is actionable
                if self._is_actionable_element(element):
                    return ElementMatch(
                        element_id=element.get('id', ''),
                        confidence=0.4,  # Lower confidence for action-based matching
                        strategy='action_based_match',
                        element_data=element,
                        matched_text=text
                    )
        
        return None
    
    def _extract_text(self, element: Dict[str, Any]) -> Optional[str]:
        """Extract text from a UI element."""
        # Try different text fields
        text_fields = ['text', 'content-desc', 'label', 'title', 'name']
        
        for field in text_fields:
            if field in element and element[field]:
                return str(element[field])
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2]
    
    def _calculate_semantic_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        """Calculate semantic similarity between two sets of keywords."""
        if not keywords1 or not keywords2:
            return 0.0
        
        # Simple Jaccard similarity
        intersection = set(keywords1) & set(keywords2)
        union = set(keywords1) | set(keywords2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _is_actionable_element(self, element: Dict[str, Any]) -> bool:
        """Check if an element is actionable (clickable, etc.)."""
        # Check for common actionable element types
        actionable_classes = ['button', 'clickable', 'tappable', 'interactive']
        
        element_class = element.get('class', '').lower()
        return any(cls in element_class for cls in actionable_classes)

# Global instance for backward compatibility
_ui_parser = UIParser()

def find_element_for_subgoal(ui_tree: List[Dict[str, Any]], subgoal: str) -> Optional[str]:
    """
    Find element for subgoal using the global UI parser instance.
    
    Args:
        ui_tree: The UI tree from the environment
        subgoal: The subgoal string
        
    Returns:
        element_id: The ID of the best matching element, or None if not found
    """
    return _ui_parser.find_element_for_subgoal(ui_tree, subgoal)
