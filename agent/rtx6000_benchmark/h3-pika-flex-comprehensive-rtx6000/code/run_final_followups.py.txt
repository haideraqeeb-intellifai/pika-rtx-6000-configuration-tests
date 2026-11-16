"""Run the final matched MPS control and combined loading/export configuration."""
from .run_mps_control_followup import main as mps_control
from .run_combined_followup import main as combined
if __name__=='__main__':
    mps_control()
    combined()
    print('Final follow-ups complete',flush=True)
