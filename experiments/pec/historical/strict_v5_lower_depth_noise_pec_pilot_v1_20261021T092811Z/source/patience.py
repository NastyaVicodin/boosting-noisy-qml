"""Same strict-best-loss, consecutive-epoch rule as boosting convergence.py."""
import math

class Patience100:
    def __init__(self):
        self.epoch=0;self.best_epoch=0;self.best_loss=None;self.bad_epochs=0
    def update(self,loss,epoch):
        assert epoch==self.epoch+1
        self.epoch=epoch;value=float(loss)
        if not math.isfinite(value):raise FloatingPointError('Nonfinite epoch training loss')
        improved=self.best_loss is None or value<self.best_loss
        if improved:self.best_loss=value;self.best_epoch=epoch;self.bad_epochs=0
        else:self.bad_epochs+=1
        return dict(epoch=epoch,epoch_loss=value,improved=improved,best_epoch=self.best_epoch,best_loss=self.best_loss,
                    epochs_without_improvement=self.bad_epochs,should_stop=self.bad_epochs>=100)

def scan_updates(records):
    tracker=Patience100();history=[]
    for i in range(len(records)//24):
        batch=records[i*24:(i+1)*24]
        assert [v['update'] for v in batch]==list(range(i*24+1,(i+1)*24+1))
        decision=tracker.update(sum(float(v['loss']) for v in batch)/24,i+1);history.append(decision)
        if decision['should_stop']:break
    return history

def tracker_from_history(history):
    tracker=Patience100()
    for v in history:tracker.update(v["epoch_loss"],v["epoch"])
    return tracker
