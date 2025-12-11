"""
Assessment Module for Surgical Training System
==============================================

This module evaluates trainee performance during surgical procedures
using weighted scoring algorithms and statistical analysis.

Mathematical Concepts Implemented:
---------------------------------
1. **Statistics**:
   - Weighted scoring with penalties
   - Accuracy calculation: Accuracy = (correct / total) × 100
   - Time-based penalties

2. **Scoring Algorithm**:
   Final Score = base_score - mistake_penalty - time_penalty
   Where:
   - mistake_penalty = 5 × number_of_mistakes
   - time_penalty = (time / 600) × 10

3. **Damage Assessment**:
   score -= damage_amount × 0.5
   Score clamped to [0, 100]

Author: Medical Simulation Project Team
Style: PEP-8 Compliant
"""

import time
from typing import Dict, List, Any, Tuple, Optional
from procedure_scripting import ProcedureScript, ProcedureAction


class AssessmentModule:
    """
    Evaluates surgical procedure performance using weighted scoring.
    
    Mathematical Model:
    -------------------
    The scoring system uses a multi-factor penalty model:
    
    1. **Base Score**: Starts at 100 (perfect score)
    
    2. **Action Penalty**:
       score = score - 5  (for each incorrect action)
    
    3. **Damage Penalty**:
       score = score - (damage_amount × 0.5)
    
    4. **Time Penalty** (applied at finalization):
       time_penalty = (total_time / 600) × 10
       final_score = score - time_penalty
       
       This penalizes 10 points for every 10 minutes.
    
    5. **Score Bounds**:
       final_score = max(0, final_score)  (clamped to non-negative)
    
    Attributes:
        script: The procedure script being assessed
        start_time: Timestamp when procedure started
        action_log: List of all actions taken
        score: Current score (0-100)
        mistake_count: Number of mistakes made
        damage_log: Dictionary tracking organ damage
    """
    
    def __init__(self, procedure_script: ProcedureScript) -> None:
        """
        Initialize assessment module for a procedure.
        
        Args:
            procedure_script: The procedure to assess
        """
        self.script: ProcedureScript = procedure_script
        self.start_time: float = time.time()
        self.action_log: List[Dict[str, Any]] = []
        self.score: float = 100  # Start with perfect score
        self.mistake_count: int = 0
        self.damage_log: Dict[str, float] = {}  # Tracks damage to organs

    def log_action(
        self, 
        action_data: Dict[str, Any], 
        is_correct: bool, 
        feedback: str = ""
    ) -> None:
        """
        Record an action in the assessment log.
        
        Mathematical Operation:
        ----------------------
        If action is incorrect:
            mistake_count = mistake_count + 1
            score = score - 5  (penalty per mistake)
        
        Args:
            action_data: Dictionary containing action details
            is_correct: Whether the action was correct
            feedback: Feedback message for the action
        """
        log_entry: Dict[str, Any] = {
            'timestamp': time.time() - self.start_time,
            'action': action_data['action'].name,
            'data': action_data,
            'is_correct': is_correct,
            'feedback': feedback,
            'step_id': self.script.current_step_index + 1
        }
        self.action_log.append(log_entry)
        
        # Apply mistake penalty
        MISTAKE_PENALTY = 5
        if not is_correct:
            self.mistake_count += 1
            self.score -= MISTAKE_PENALTY

    def check_action(self, action_data):
        current_step = self.script.get_current_step()
        if not current_step:
            self.log_action(action_data, False, "Procedure complete. No further actions required.")
            return False, "Procedure complete. No further actions required."

        is_correct = current_step.check_completion(action_data)
        
        if is_correct:
            feedback = f"Correct! Proceeding to Step {current_step.step_id + 1}."
            self.log_action(action_data, True, feedback)
            self.script.advance_step()
        else:
            feedback = f"Incorrect action for Step {current_step.step_id}. Required: {current_step.required_action.name}."
            self.log_action(action_data, False, feedback)
            
        return is_correct, feedback

    def log_damage(self, organ_name: str, damage_amount: float) -> None:
        """
        Record damage to an organ and apply score penalty.
        
        Mathematical Formula:
        --------------------
        cumulative_damage[organ] = cumulative_damage[organ] + damage_amount
        score = score - (damage_amount × 0.5)
        score = max(0, score)  (clamp to non-negative)
        
        Args:
            organ_name: Name of the damaged organ
            damage_amount: Amount of damage (percentage)
        """
        DAMAGE_PENALTY_FACTOR = 0.5
        
        if damage_amount > 0:
            # Accumulate damage for the organ
            self.damage_log[organ_name] = self.damage_log.get(organ_name, 0) + damage_amount
            
            # Apply penalty proportional to damage
            self.score -= damage_amount * DAMAGE_PENALTY_FACTOR
            
            # Clamp score to non-negative
            self.score = max(0, self.score)
            
            self.log_action(
                {'action': ProcedureAction.DISSECTION, 'target': organ_name}, 
                False, 
                f"Caution: Caused {damage_amount}% damage to {organ_name}."
            )

    def finalize_assessment(self) -> Dict[str, Any]:
        """
        Generate final assessment report with time-based penalties.
        
        Mathematical Formula (Final Score Calculation):
        -----------------------------------------------
        time_penalty = (total_time / 600) × 10
        
        This means:
        - 10 minutes → 10 points penalty
        - 20 minutes → 20 points penalty
        - etc.
        
        final_score = round(base_score - time_penalty, 1)
        final_score = max(0, final_score)  (clamp to non-negative)
        
        Returns:
            Dictionary containing:
            - procedure_name: Name of the procedure
            - completed: Whether all steps were completed
            - final_score: Calculated final score (0-100)
            - total_time_seconds: Time taken in seconds
            - mistake_count: Number of mistakes made
            - damage_summary: Dictionary of organ damage
            - action_log: List of all recorded actions
        """
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Time penalty: 10 points per 10 minutes (600 seconds)
        TIME_PENALTY_RATE = 10  # points per 600 seconds
        TIME_PENALTY_PERIOD = 600  # seconds
        
        time_penalty = (total_time / TIME_PENALTY_PERIOD) * TIME_PENALTY_RATE
        final_score = round(self.score - time_penalty, 1)
        final_score = max(0, final_score)  # Clamp to non-negative
        
        report: Dict[str, Any] = {
            "procedure_name": self.script.name,
            "completed": self.script.get_current_step() is None,
            "final_score": final_score,
            "total_time_seconds": round(total_time, 2),
            "mistake_count": self.mistake_count,
            "damage_summary": self.damage_log,
            "action_log": self.action_log
        }
        return report

# --- Example Usage ---
if __name__ == '__main__':
    from procedure_scripting import create_appendectomy_script

    script = create_appendectomy_script()
    assessment = AssessmentModule(script)
    
    print(f"Starting Procedure: {script.name}")
    
    # Step 1: Identify Organ (Correct)
    action_data_1 = {'action': ProcedureAction.IDENTIFY_ORGAN, 'target': 'appendix'}
    is_correct, feedback = assessment.check_action(action_data_1)
    print(f"Action 1: Correct={is_correct}, Feedback='{feedback}'")
    
    # Simulate accidental damage
    assessment.log_damage('intestine', 10)
    
    # Step 2: Incision (Incorrect action type)
    action_data_2_fail = {'action': ProcedureAction.SUTURE, 'target': 'skin'}
    is_correct, feedback = assessment.check_action(action_data_2_fail)
    print(f"Action 2 (Fail): Correct={is_correct}, Feedback='{feedback}'")
    
    # Step 2: Incision (Correct)
    action_data_2_success = {'action': ProcedureAction.INCISION, 'target': 'skin', 'tool': 'scalpel'}
    is_correct, feedback = assessment.check_action(action_data_2_success)
    print(f"Action 2 (Success): Correct={is_correct}, Feedback='{feedback}'")

    # Skip remaining steps for brevity
    while script.get_current_step():
        script.advance_step()
        
    report = assessment.finalize_assessment()
    
    print("\n--- Final Assessment Report ---")
    print(f"Procedure: {report['procedure_name']}")
    print(f"Completed: {report['completed']}")
    print(f"Final Score: {report['final_score']}")
    print(f"Total Time: {report['total_time_seconds']}s")
    print(f"Mistakes: {report['mistake_count']}")
    print(f"Damage: {report['damage_summary']}")
    # print("\nAction Log:")
    # for log in report['action_log']:
    #     print(f"  [{log['timestamp']:.2f}s] Step {log['step_id']} - {log['action']}: {log['feedback']}")
