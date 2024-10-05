import torch
import torch.nn.functional as F

from simplemodel import SimpleModel
from config import Parameter

class categoricalpolicy:
    def sampled_action(action_logits):
        action_probs  = F.softmax(action_logits,dim=-1)
        sampled_action = torch.multinomial(action_probs,1)
        return action_probs, sampled_action
    def log_likelihood(action_probs, sampled_action, batch_size):
        log_likelihood = torch.log(torch.diagonal(action_probs[range(batch_size), sampled_action ]))
        return log_likelihood

if __name__ == "__main__":

    cate_parameter = Parameter()
    policy_model = SimpleModel(cate_parameter)
    state         = torch.randn((cate_parameter.batch_size,cate_parameter.state_size))
    action_logits = policy_model.forward(x=state)
    action_probs  = F.softmax(action_logits,dim=-1)
    sampled_action = torch.multinomial(action_probs,1)

    log_likelihood = torch.log(torch.diagonal(action_probs[range(cate_parameter.batch_size), sampled_action ]))
    print(log_likelihood)
