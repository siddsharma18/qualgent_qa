from typing import Dict, Any, Optional, List, Tuple, Set
import difflib
import re
import time
import logging
from dataclasses import dataclass
from enum import Enum
from utils.logger import QALogger

class VerificationStatus(Enum):
    """Verification status enumeration."""
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class VerificationResult:
    """Result of verification with detailed information."""
    status: VerificationStatus
    confidence: float
    reason: str
    needs_replan: bool
    verification_time: float
    strategies_used: List[str]
    ui_changes_detected: Dict[str, Any]
    element_matches: List[Dict[str, Any]]
    stability_score: float = 0.0
    false_positive_risk: float = 0.0
    alternative_interpretations: List[str] = None

    def __post_init__(self):
        if self.alternative_interpretations is None:
            self.alternative_interpretations = []

class VerifierAgent:
    """
    A robust verifier agent that uses multiple strategies to verify whether
    a subgoal has been successfully completed.
    """

    def __init__(self, 
                 logger: Optional[QALogger] = None,
                 min_change_ratio: float = 0.1,
                 fuzzy_threshold: float = 0.7,
                 max_verification_time: float = 10.0,
                 enable_advanced_analysis: bool = True,
                 enable_state_tracking: bool = True,
                 min_confidence: float = 0.5):
        """
        Initialize the verifier agent.
        
        Args:
            logger: Optional logger for recording verification activities
            min_change_ratio: Minimum ratio of UI changes to consider significant
            fuzzy_threshold: Threshold for fuzzy string matching
            max_verification_time: Maximum time to spend on verification
            enable_advanced_analysis: Whether to use advanced UI analysis
            enable_state_tracking: Whether to track UI state changes
            min_confidence: Minimum confidence threshold for verification
        """
        self.logger = logger or QALogger()
        self.min_change_ratio = min_change_ratio
        self.fuzzy_threshold = fuzzy_threshold
        self.max_verification_time = max_verification_time
        self.enable_advanced_analysis = enable_advanced_analysis
        self.enable_state_tracking = enable_state_tracking
        self.min_confidence = min_confidence
        
        # Track verification statistics
        self.stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'uncertain_verifications': 0,
            'average_verification_time': 0.0,
            'strategy_success_rates': {}
        }
        
        # UI state tracking
        self.state_history = []
        
        # Common verification patterns
        self.verification_patterns = {
            'wifi': {
                'keywords': ['wifi', 'wi-fi', 'wireless', 'network'],
                'toggle_states': ['on', 'off', 'enabled', 'disabled'],
                'expected_changes': ['switch', 'toggle', 'button']
            },
            'bluetooth': {
                'keywords': ['bluetooth', 'bt', 'bluetooth'],
                'toggle_states': ['on', 'off', 'enabled', 'disabled'],
                'expected_changes': ['switch', 'toggle', 'button']
            },
            'brightness': {
                'keywords': ['brightness', 'screen brightness', 'display brightness'],
                'expected_changes': ['slider', 'seekbar', 'progress']
            },
            'volume': {
                'keywords': ['volume', 'sound', 'audio'],
                'expected_changes': ['slider', 'seekbar', 'progress']
            },
            'settings': {
                'keywords': ['settings', 'preferences', 'options'],
                'expected_changes': ['activity', 'screen', 'page']
            }
        }

    def verify(self,
               subgoal: str,
               prev_obs: Dict[str, Any],
               curr_obs: Dict[str, Any]) -> VerificationResult:
        """
        Verify whether a subgoal has been successfully completed using multiple strategies.
        
        Args:
            subgoal: Subgoal description (e.g., "Turn Wi-Fi off")
            prev_obs: Observation before execution
            curr_obs: Observation after execution
            
        Returns:
            VerificationResult: Detailed verification result
        """
        start_time = time.time()
        self.stats['total_verifications'] += 1
        
        # Log verification start
        self.logger.info("Verifier", "Verification started", 
                        subgoal=subgoal,
                        prev_obs_keys=list(prev_obs.keys()) if prev_obs else [],
                        curr_obs_keys=list(curr_obs.keys()) if curr_obs else [])
        
        # Validate inputs
        if not self._validate_observations(prev_obs, curr_obs):
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.0,
                reason="Invalid or missing observations",
                needs_replan=True,
                verification_time=time.time() - start_time,
                strategies_used=[],
                ui_changes_detected={},
                element_matches=[]
            )
        
        # Extract UI trees
        prev_tree = prev_obs.get("ui_tree", [])
        curr_tree = curr_obs.get("ui_tree", [])
        
        # Track state if enabled
        if self.enable_state_tracking:
            self._track_state_change(prev_tree, curr_tree, subgoal)
        
        # Apply multiple verification strategies
        results = []
        strategies_used = []
        
        verification_strategies = [
            self._ui_change_verification,
            self._subgoal_presence_verification,
            self._state_transition_verification,
            self._element_interaction_verification,
            self._semantic_verification
        ]
        
        for strategy in verification_strategies:
            try:
                result = strategy(subgoal, prev_tree, curr_tree)
                if result:
                    results.append(result)
                    strategies_used.append(strategy.__name__)
            except Exception as e:
                self.logger.warning("Verifier", "Strategy failed", 
                                  strategy=strategy.__name__,
                                  error=str(e))
        
        # Combine results and make final decision
        final_result = self._combine_verification_results(
            results, strategies_used, subgoal, start_time
        )
        
        # Update statistics
        self._update_stats(final_result, time.time() - start_time)
        
        return final_result

    def _validate_observations(self, prev_obs: Dict[str, Any], curr_obs: Dict[str, Any]) -> bool:
        """Validate that observations contain required data."""
        if not prev_obs or not curr_obs:
            return False
        
        if "ui_tree" not in prev_obs or "ui_tree" not in curr_obs:
            return False
        
        if not isinstance(prev_obs["ui_tree"], list) or not isinstance(curr_obs["ui_tree"], list):
            return False
        
        return True

    def _ui_change_verification(self, subgoal: str, prev_tree: List[Dict], curr_tree: List[Dict]) -> Optional[Dict[str, Any]]:
        """Verify based on UI tree changes."""
        if not prev_tree or not curr_tree:
            return None
        
        # Calculate UI change metrics
        prev_ids = set(e.get("id", "") for e in prev_tree)
        curr_ids = set(e.get("id", "") for e in curr_tree)
        
        # Element addition/removal
        added_elements = curr_ids - prev_ids
        removed_elements = prev_ids - curr_ids
        
        # Element state changes
        state_changes = self._detect_state_changes(prev_tree, curr_tree)
        
        # Calculate change ratio
        total_elements = len(prev_ids | curr_ids)
        change_ratio = len(added_elements | removed_elements) / max(total_elements, 1)
        
        # Determine if changes are significant
        is_significant = change_ratio >= self.min_change_ratio or len(state_changes) > 0
        
        confidence = min(change_ratio * 2, 1.0) if is_significant else 0.0
        
        return {
            'strategy': 'ui_change_verification',
            'confidence': confidence,
            'is_significant': is_significant,
            'change_ratio': change_ratio,
            'added_elements': list(added_elements),
            'removed_elements': list(removed_elements),
            'state_changes': state_changes
        }

    def _subgoal_presence_verification(self, subgoal: str, prev_tree: List[Dict], curr_tree: List[Dict]) -> Optional[Dict[str, Any]]:
        """Verify based on subgoal text presence in UI."""
        subgoal_lower = subgoal.lower()
        
        # Extract key terms from subgoal
        key_terms = self._extract_key_terms(subgoal_lower)
        
        # Find matching elements in current UI
        matches = []
        for element in curr_tree:
            element_text = self._extract_element_text(element)
            if element_text:
                match_score = self._calculate_text_match_score(key_terms, element_text.lower())
                if match_score > 0:
                    matches.append({
                        'element_id': element.get('id', ''),
                        'text': element_text,
                        'match_score': match_score
                    })
        
        # Calculate confidence based on best matches
        if matches:
            best_match = max(matches, key=lambda x: x['match_score'])
            confidence = best_match['match_score']
        else:
            confidence = 0.0
        
        return {
            'strategy': 'subgoal_presence_verification',
            'confidence': confidence,
            'matches': matches,
            'key_terms': key_terms
        }

    def _state_transition_verification(self, subgoal: str, prev_tree: List[Dict], curr_tree: List[Dict]) -> Optional[Dict[str, Any]]:
        """Verify based on expected state transitions."""
        subgoal_lower = subgoal.lower()
        
        # Identify the type of action from subgoal
        action_type = self._identify_action_type(subgoal_lower)
        
        if not action_type:
            return None
        
        # Get expected state changes for this action type
        expected_changes = self._get_expected_state_changes(action_type, subgoal_lower)
        
        # Check if expected changes occurred
        actual_changes = self._detect_state_changes(prev_tree, curr_tree)
        
        # Calculate match between expected and actual changes
        matches = 0
        total_expected = len(expected_changes)
        
        for expected_change in expected_changes:
            for actual_change in actual_changes:
                if self._fuzzy_match(expected_change, actual_change):
                    matches += 1
                    break
        
        confidence = matches / max(total_expected, 1) if total_expected > 0 else 0.0
        
        return {
            'strategy': 'state_transition_verification',
            'confidence': confidence,
            'action_type': action_type,
            'expected_changes': expected_changes,
            'actual_changes': actual_changes,
            'matches': matches
        }

    def _element_interaction_verification(self, subgoal: str, prev_tree: List[Dict], curr_tree: List[Dict]) -> Optional[Dict[str, Any]]:
        """Verify based on element interaction patterns."""
        # Look for interactive elements that might have been affected
        interactive_elements = []
        
        for element in curr_tree:
            if self._is_interactive_element(element):
                interactive_elements.append({
                    'id': element.get('id', ''),
                    'text': self._extract_element_text(element),
                    'class': element.get('class', ''),
                    'clickable': element.get('clickable', False)
                })
        
        # Check if any interactive elements match the subgoal
        matches = []
        for element in interactive_elements:
            if element['text']:
                match_score = self._calculate_text_match_score(
                    self._extract_key_terms(subgoal.lower()),
                    element['text'].lower()
                )
                if match_score > 0.3:  # Lower threshold for interactive elements
                    matches.append({
                        'element': element,
                        'match_score': match_score
                    })
        
        confidence = max([m['match_score'] for m in matches]) if matches else 0.0
        
        return {
            'strategy': 'element_interaction_verification',
            'confidence': confidence,
            'interactive_elements': interactive_elements,
            'matches': matches
        }

    def _semantic_verification(self, subgoal: str, prev_tree: List[Dict], curr_tree: List[Dict]) -> Optional[Dict[str, Any]]:
        """Verify based on semantic understanding of the subgoal."""
        if not self.enable_advanced_analysis:
            return None
        
        # Extract semantic meaning from subgoal
        semantic_info = self._extract_semantic_info(subgoal)
        
        if not semantic_info:
            return None
        
        # Check if current UI state reflects the expected semantic state
        semantic_matches = []
        
        for element in curr_tree:
            element_text = self._extract_element_text(element)
            if element_text:
                semantic_score = self._calculate_semantic_similarity(
                    semantic_info, element_text.lower()
                )
                if semantic_score > 0.4:
                    semantic_matches.append({
                        'element_id': element.get('id', ''),
                        'text': element_text,
                        'semantic_score': semantic_score
                    })
        
        confidence = max([m['semantic_score'] for m in semantic_matches]) if semantic_matches else 0.0
        
        return {
            'strategy': 'semantic_verification',
            'confidence': confidence,
            'semantic_info': semantic_info,
            'semantic_matches': semantic_matches
        }

    def _combine_verification_results(self, 
                                    results: List[Dict[str, Any]], 
                                    strategies_used: List[str],
                                    subgoal: str,
                                    start_time: float) -> VerificationResult:
        """Combine multiple verification results into a final decision."""
        if not results:
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                confidence=0.0,
                reason="No verification strategies produced results",
                needs_replan=True,
                verification_time=time.time() - start_time,
                strategies_used=strategies_used,
                ui_changes_detected={},
                element_matches=[]
            )
        
        # Calculate weighted confidence
        total_confidence = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = self._get_strategy_weight(result['strategy'])
            total_confidence += result['confidence'] * weight
            total_weight += weight
        
        final_confidence = total_confidence / max(total_weight, 1)
        
        # Determine status based on confidence
        if final_confidence >= self.min_confidence:
            status = VerificationStatus.PASS
            reason = f"High confidence verification ({final_confidence:.2f})"
            needs_replan = False
        elif final_confidence >= self.min_confidence * 0.6:
            status = VerificationStatus.UNCERTAIN
            reason = f"Moderate confidence verification ({final_confidence:.2f})"
            needs_replan = False
        else:
            status = VerificationStatus.FAIL
            reason = f"Low confidence verification ({final_confidence:.2f})"
            needs_replan = True
        
        # Collect UI changes and element matches
        ui_changes = {}
        element_matches = []
        
        for result in results:
            if 'change_ratio' in result:
                ui_changes['change_ratio'] = result['change_ratio']
            if 'matches' in result and isinstance(result['matches'], list):
                element_matches.extend(result['matches'])
        
        return VerificationResult(
            status=status,
            confidence=final_confidence,
            reason=reason,
            needs_replan=needs_replan,
            verification_time=time.time() - start_time,
            strategies_used=strategies_used,
            ui_changes_detected=ui_changes,
            element_matches=element_matches
        )

    def _detect_state_changes(self, prev_tree: List[Dict], curr_tree: List[Dict]) -> List[str]:
        """Detect state changes between UI trees."""
        changes = []
        
        # Create lookup for previous elements
        prev_elements = {e.get('id', ''): e for e in prev_tree}
        
        for curr_element in curr_tree:
            element_id = curr_element.get('id', '')
            if element_id in prev_elements:
                prev_element = prev_elements[element_id]
                
                # Check for text changes
                prev_text = self._extract_element_text(prev_element)
                curr_text = self._extract_element_text(curr_element)
                
                if prev_text != curr_text:
                    changes.append(f"text_change:{element_id}")
                
                # Check for enabled/disabled state changes
                prev_enabled = prev_element.get('enabled', True)
                curr_enabled = curr_element.get('enabled', True)
                
                if prev_enabled != curr_enabled:
                    changes.append(f"enabled_change:{element_id}")
                
                # Check for checked state changes
                prev_checked = prev_element.get('checked', False)
                curr_checked = curr_element.get('checked', False)
                
                if prev_checked != curr_checked:
                    changes.append(f"checked_change:{element_id}")
        
        return changes

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Remove common words and extract meaningful terms
        common_words = {'turn', 'on', 'off', 'enable', 'disable', 'open', 'close', 'click', 'tap', 'press', 'the', 'and', 'or', 'to', 'in', 'of', 'for', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        meaningful_terms = [word for word in words if len(word) > 2 and word not in common_words]
        
        # If no meaningful terms found, include some common action words
        if not meaningful_terms:
            action_words = [word for word in words if word in ['wifi', 'bluetooth', 'settings', 'brightness', 'volume']]
            meaningful_terms.extend(action_words)
        
        return meaningful_terms

    def _extract_element_text(self, element: Dict[str, Any]) -> Optional[str]:
        """Extract text from UI element."""
        text_fields = ['text', 'content-desc', 'label', 'title', 'name']
        for field in text_fields:
            if field in element and element[field]:
                return str(element[field])
        return None

    def _calculate_text_match_score(self, key_terms: List[str], text: str) -> float:
        """Calculate match score between key terms and text."""
        if not key_terms or not text:
            return 0.0
        
        matches = 0
        for term in key_terms:
            if term in text:
                matches += 1
        
        return matches / len(key_terms)

    def _fuzzy_match(self, s1: str, s2: str) -> bool:
        """Check if two strings match using fuzzy matching."""
        ratio = difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
        return ratio >= self.fuzzy_threshold

    def _identify_action_type(self, subgoal: str) -> Optional[str]:
        """Identify the type of action from subgoal."""
        action_patterns = {
            'toggle': ['turn on', 'turn off', 'enable', 'disable', 'toggle'],
            'navigate': ['open', 'go to', 'navigate to', 'enter'],
            'input': ['type', 'enter', 'input', 'write'],
            'scroll': ['scroll', 'swipe', 'move'],
            'back': ['go back', 'return', 'previous']
        }
        
        for action_type, patterns in action_patterns.items():
            if any(pattern in subgoal for pattern in patterns):
                return action_type
        
        return None

    def _get_expected_state_changes(self, action_type: str, subgoal: str) -> List[str]:
        """Get expected state changes for an action type."""
        if action_type == 'toggle':
            return ['enabled_change', 'checked_change']
        elif action_type == 'navigate':
            return ['text_change', 'activity_change']
        elif action_type == 'input':
            return ['text_change']
        elif action_type == 'scroll':
            return ['position_change']
        else:
            return []

    def _is_interactive_element(self, element: Dict[str, Any]) -> bool:
        """Check if element is interactive."""
        interactive_classes = ['button', 'clickable', 'tappable', 'interactive']
        element_class = element.get('class', '').lower()
        return any(cls in element_class for cls in interactive_classes) or element.get('clickable', False)

    def _extract_semantic_info(self, subgoal: str) -> Optional[Dict[str, Any]]:
        """Extract semantic information from subgoal."""
        # Simple semantic extraction - can be enhanced with NLP
        subgoal_lower = subgoal.lower()
        
        for category, patterns in self.verification_patterns.items():
            if any(keyword in subgoal_lower for keyword in patterns['keywords']):
                return {
                    'category': category,
                    'keywords': patterns['keywords'],
                    'expected_changes': patterns.get('expected_changes', [])
                }
        
        return None

    def _calculate_semantic_similarity(self, semantic_info: Dict[str, Any], text: str) -> float:
        """Calculate semantic similarity between semantic info and text."""
        if not semantic_info or not text:
            return 0.0
        
        # Check for keyword matches
        keyword_matches = sum(1 for keyword in semantic_info['keywords'] if keyword in text)
        return keyword_matches / len(semantic_info['keywords'])

    def _get_strategy_weight(self, strategy_name: str) -> float:
        """Get weight for a verification strategy."""
        weights = {
            'ui_change_verification': 1.0,
            'subgoal_presence_verification': 0.8,
            'state_transition_verification': 0.9,
            'element_interaction_verification': 0.7,
            'semantic_verification': 0.6
        }
        return weights.get(strategy_name, 0.5)

    def _track_state_change(self, prev_tree: List[Dict], curr_tree: List[Dict], subgoal: str):
        """Track UI state changes for analysis."""
        state_info = {
            'timestamp': time.time(),
            'subgoal': subgoal,
            'prev_element_count': len(prev_tree),
            'curr_element_count': len(curr_tree),
            'changes': self._detect_state_changes(prev_tree, curr_tree)
        }
        self.state_history.append(state_info)
        
        # Keep only recent history
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]

    def _update_stats(self, result: VerificationResult, verification_time: float):
        """Update verification statistics."""
        if result.status == VerificationStatus.PASS:
            self.stats['successful_verifications'] += 1
        elif result.status == VerificationStatus.FAIL:
            self.stats['failed_verifications'] += 1
        elif result.status == VerificationStatus.UNCERTAIN:
            self.stats['uncertain_verifications'] += 1
        
        # Update average verification time
        total_verifications = (self.stats['successful_verifications'] + 
                             self.stats['failed_verifications'] + 
                             self.stats['uncertain_verifications'])
        
        if total_verifications > 0:
            self.stats['average_verification_time'] = (
                (self.stats['average_verification_time'] * (total_verifications - 1) + verification_time) 
                / total_verifications
            )
        
        # Update strategy success rates
        for strategy in result.strategies_used:
            if strategy not in self.stats['strategy_success_rates']:
                self.stats['strategy_success_rates'][strategy] = {'success': 0, 'total': 0}
            
            self.stats['strategy_success_rates'][strategy]['total'] += 1
            if result.status == VerificationStatus.PASS:
                self.stats['strategy_success_rates'][strategy]['success'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset verification statistics."""
        self.stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'uncertain_verifications': 0,
            'average_verification_time': 0.0,
            'strategy_success_rates': {}
        }

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get UI state change history."""
        return self.state_history.copy()