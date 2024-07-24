import torch
import torch.nn as nn
import torch.nn.functional as F
from hypnettorch.hnets import HyperNetInterface


def Regularizer_LSTM(model, alpha=1e-5, l1_ratio=0.5):
    """
    Implement an L1-L2 penalty on the norm of the model weights.

    model: CausalTemporalLSTM instance
    alpha: scaling parameter for the regularization.
    l1_ratio: mixing parameter between L1 and L2 loss.

    Returns:
    reg: regularization term
    """
    w_t = model.lstm.weight_ih_l0
    w_l_1 = model.linear.weight
   

    l1_loss = w_t.abs().sum() + w_l_1.abs().sum()
    l2_loss = w_t.pow(2.0).sum() + w_l_1.pow(2.0).sum() 

    reg = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg = alpha * reg

    return reg


def Regularizer_RNN(model, alpha=1e-5, l1_ratio=0.5):
    """
    Implement an L1-L2 penalty on the norm of the model weights.

    model: CausalTemporalLSTM instance
    alpha: scaling parameter for the regularization.
    l1_ratio: mixing parameter between L1 and L2 loss.

    Returns:
    reg: regularization term
    """
    w_t = model.rnn.weight_ih_l0
    w_l = model.linear.weight
 

    l1_loss = w_t.abs().sum() + w_l.abs().sum() 
    l2_loss = w_t.pow(2.0).sum() + w_l.pow(2.0).sum() 

    reg = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg = alpha * reg

    return reg #.item()

###############################################################
### Regularization function for hypernetworks:
###############################################################

##### Regularization for the current task
def reg_hnet(weights, alpha, l1_ratio):
    
    """
    Implement an L1-L2 penalty on the norm of the model weights.

    model: MLP
    alpha: scaling parameter for the regularization.
    l1_ratio: mixing parameter between L1 and L2 loss.

    Returns:
    reg: regularization term
    """
    l1_loss = 0
    l2_loss = 0

    # Accumulate L1 and L2 losses for weight matrices in the model
    

    weights_ =  [i for i in weights if len(i.shape)==2]

    for weight_tensor in weights_[:2]:
        l1_loss += torch.sum(torch.abs(weight_tensor))
        l2_loss += torch.sum(weight_tensor.pow(2))

    reg = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg = alpha * reg

    return reg

def reg_hnet_allweights(weights, alpha, l1_ratio):
    """
    Implement an L1-L2 penalty on the norm of the model weights.

    model: MLP
    alpha: scaling parameter for the regularization.
    l1_ratio: mixing parameter between L1 and L2 loss.

    Returns:
    reg: regularization term
    """
    l1_loss = 0
    l2_loss = 0

    # Accumulate L1 and L2 losses for weight matrices in the model
    for weight_tensor in weights:
        l1_loss += torch.sum(torch.abs(weight_tensor))
        l2_loss += torch.sum(weight_tensor.pow(2))

    reg = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg = alpha * reg

    # Accumulate L1 and L2 losses for weight matrices in the model
    for weight_tensor in weights:
        l1_loss += torch.sum(torch.abs(weight_tensor))
        l2_loss += torch.sum(weight_tensor.pow(2))

    reg_item = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg_item = alpha * reg_item

    return reg

# Anirudh's Function to get the weights generated with the model using previous conditions
def get_current_targets(task_id, hnet):
    r"""For all :math:`j < \text{task\_id}`, compute the output of the
    hypernetwork. This output will be detached from the graph before being added
    to the return list of this function.

    Note, if these targets don't change during training, it would be more memory
    efficient to store the weights :math:`\theta^*` of the hypernetwork (which
    is a fixed amount of memory compared to the variable number of tasks).
    Though, it is more computationally expensive to recompute
    :math:`h(c_j, \theta^*)` for all :math:`j < \text{task\_id}` everytime the
    target is needed.

    Note, this function sets the hypernet temporarily in eval mode. No gradients
    are computed.

    See argument ``targets`` of :func:`calc_fix_target_reg` for a use-case of
    this function.

    Args:
        task_id (int): The ID of the current task.
        hnet: An instance of the hypernetwork before learning a new task
            (i.e., the hypernetwork has the weights :math:`\theta^*` necessary
            to compute the targets).

    Returns:
        An empty list, if ``task_id`` is ``0``. Otherwise, a list of
        ``task_id-1`` targets. These targets can be passed to the function
        :func:`calc_fix_target_reg` while training on the new task.
    """
    # We temporarily switch to eval mode for target computation (e.g., to get
    # rid of training stochasticities such as dropout).
    hnet_mode = hnet.training
    hnet.eval()

    ret = []

    with torch.no_grad():
        W = hnet.forward(cond_id=list(range(task_id)), ret_format="sequential")
        ret = [[p.detach() for p in W_tid] for W_tid in W]

    hnet.train(mode=hnet_mode)

    return ret

# Anirudh's Function to get the regularization term which depends on the previous tasks.
def calc_fix_target_reg(
    hnet,
    task_id,
    targets=None,
    dTheta=None,
    dTembs=None,
    mnet=None,
    prev_theta=None,
    prev_task_embs=None,
    batch_size=None,
    reg_scaling=None,
):
    r"""This regularizer simply restricts the output-mapping for previous
    task embeddings. I.e., for all :math:`j < \text{task\_id}` minimize:

    .. math::
        \lVert \text{target}_j - h(c_j, \theta + \Delta\theta) \rVert^2

    where :math:`c_j` is the current task embedding for task :math:`j` (and we
    assumed that ``dTheta`` was passed).

    Args:
        hnet: The hypernetwork whose output should be regularized; has to
            implement the interface
            :class:`hnets.hnet_interface.HyperNetInterface`.
        task_id (int): The ID of the current task (the one that is used to
            compute ``dTheta``).
        targets (list): A list of outputs of the hypernetwork. Each list entry
            must have the output shape as returned by the
            :meth:`hnets.hnet_interface.HyperNetInterface.forward` method of the
            ``hnet``. Note, this function doesn't detach targets. If desired,
            that should be done before calling this function.

            Also see :func:`get_current_targets`.
        dTheta (list, optional): The current direction of weight change for the
            internal (unconditional) weights of the hypernetwork evaluated on
            the task-specific loss, i.e., the weight change that would be
            applied to the unconditional parameters :math:`\theta`. This
            regularizer aims to modify this direction, such that the hypernet
            output for embeddings of previous tasks remains unaffected.
            Note, this function does not detach ``dTheta``. It is up to the
            user to decide whether dTheta should be a constant vector or
            might depend on parameters of the hypernet.

            Also see :func:`utils.optim_step.calc_delta_theta`.
        dTembs (list, optional): The current direction of weight change for the
            task embeddings of all tasks that have been learned already.
            See ``dTheta`` for details.
        mnet: Instance of the main network. Has to be provided if
            ``inds_of_out_heads`` are specified.
        prev_theta (list, optional): If given, ``prev_task_embs`` but not
            ``targets`` has to be specified. ``prev_theta`` is expected to be
            the internal unconditional weights :math:`theta` prior to learning
            the current task. Hence, it can be used to compute the targets on
            the fly (which is more memory efficient (constant memory), but more
            computationally demanding).
            The computed targets will be detached from the computational graph.
            Independent of the current hypernet mode, the targets are computed
            in ``eval`` mode.
        prev_task_embs (list, optional): If given, ``prev_theta`` but not
            ``targets`` has to be specified. ``prev_task_embs`` are the task
            embeddings (conditional parameters) of the hypernetwork.
            See docstring of ``prev_theta`` for more details.
        batch_size (int, optional): If specified, only a random subset of
            previous tasks is regularized. If the given number is bigger than
            the number of previous tasks, all previous tasks are regularized.

            Note:
                A ``batch_size`` smaller or equal to zero will be ignored
                rather than throwing an error.
        reg_scaling (list, optional): If specified, the regulariation terms for
            the different tasks are scaled arcording to the entries of this
            list.

    Returns:
        The value of the regularizer.
    """
    assert isinstance(hnet, HyperNetInterface)
    assert task_id > 0
    # FIXME We currently assume the hypernet has all parameters internally.
    # Alternatively, we could allow the parameters to be passed to us, that we
    # will then pass to the forward method.
    assert hnet.unconditional_params is not None and len(hnet.unconditional_params) > 0
    assert targets is None or len(targets) == task_id
    assert targets is None or (prev_theta is None and prev_task_embs is None)
    assert prev_theta is None or prev_task_embs is not None
    # assert prev_task_embs is None or len(prev_task_embs) >= task_id
    assert dTembs is None or len(dTembs) >= task_id
    assert reg_scaling is None or len(reg_scaling) >= task_id

    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))
    if batch_size is not None and batch_size > 0:
        if num_regs > batch_size:
            ids_to_reg = np.random.choice(
                num_regs, size=batch_size, replace=False
            ).tolist()
            num_regs = batch_size

    # FIXME Assuming all unconditional parameters are internal.
    assert len(hnet.unconditional_params) == len(hnet.unconditional_param_shapes)

    weights = dict()
    uncond_params = hnet.unconditional_params
    if dTheta is not None:
        uncond_params = hnet.add_to_uncond_params(dTheta, params=uncond_params)
    weights["uncond_weights"] = uncond_params

    if dTembs is not None:
        # FIXME That's a very unintutive solution for the user.
        assert (
            hnet.conditional_params is not None
            and len(hnet.conditional_params) == len(hnet.conditional_param_shapes)
            and len(hnet.conditional_params) == len(dTembs)
        )
        weights["cond_weights"] = hnet.add_to_uncond_params(
            dTembs, params=hnet.conditional_params
        )

    if targets is None:
        prev_weights = dict()
        prev_weights["uncond_weights"] = prev_theta
        # FIXME We just assume that `prev_task_embs` are all conditional
        # weights.
        prev_weights["cond_weights"] = prev_task_embs

    reg = 0

    for i in ids_to_reg:
        weights_predicted = hnet.forward(cond_id=i, weights=weights)
        #print('Computed weights for the regularizer from task ', i )

        if targets is not None:
            target = targets[i]
        else:
            # Compute targets in eval mode!
            hnet_mode = hnet.training
            hnet.eval()

            # Compute target on the fly using previous hnet.
            with torch.no_grad():
                target = hnet.forward(cond_id=i, weights=prev_weights)
            target = [d.detach().clone() for d in target]

            hnet.train(mode=hnet_mode)

        # Regularize all weights of the main network.
        W_target = torch.cat([w.view(-1) for w in target])
        W_predicted = torch.cat([w.view(-1) for w in weights_predicted])

        reg_i = (W_target - W_predicted).pow(2).sum()

        if reg_scaling is not None:
            reg += reg_scaling[i] * reg_i
        else:
            reg += reg_i

    return reg / num_regs

