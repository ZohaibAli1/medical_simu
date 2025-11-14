import time
from procedure_scripting import ProcedureScript, ProcedureAction

class AssessmentModule:
    def __init__(self, procedure_script: ProcedureScript):
        self.script = procedure_script
        self.start_time = time.time()
        self.action_log = []
        self.score = 100  # Start with a perfect score
        self.mistake_count = 0
        self.damage_log = {} # Tracks damage to organs

    def log_action(self, action_data, is_correct, feedback=""):
        log_entry = {
            'timestamp': time.time() - self.start_time,
            'action': action_data['action'].name,
            'data': action_data,
            'is_correct': is_correct,
            'feedback': feedback,
            'step_id': self.script.current_step_index + 1
        }
        self.action_log.append(log_entry)
        
        if not is_correct:
            self.mistake_count += 1
            self.score -= 5 # Penalty for incorrect action

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

    def log_damage(self, organ_name, damage_amount):
        if damage_amount > 0:
            self.damage_log[organ_name] = self.damage_log.get(organ_name, 0) + damage_amount
            self.score -= damage_amount * 0.5 # Smaller penalty for minor damage
            
            # Ensure score doesn't drop below zero
            self.score = max(0, self.score)
            
            self.log_action({'action': ProcedureAction.DISSECTION, 'target': organ_name}, 
                            False, 
                            f"Caution: Caused {damage_amount}% damage to {organ_name}."
                           )

    def finalize_assessment(self):
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Final score calculation
        final_score = round(self.score - (total_time / 600) * 10, 1) # Time penalty (10 points per 10 minutes)
        final_score = max(0, final_score)
        
        report = {
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
