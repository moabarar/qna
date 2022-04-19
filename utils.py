import jax
import jax.numpy as jnp
from flax.training import checkpoints
from jax import tree_util


def init_from_pretrained(state, config):
  assert config.get('pretrained_path', None) != None
  pretrained_state = checkpoints.restore_checkpoint(config.pretrained_path, None)
  pretrained_params = pretrained_state['params']
  pretrained_params_leaves, _ = tree_util.tree_flatten(pretrained_params)

  current_params = state.params
  current_params_leaves, current_params_structure = tree_util.tree_flatten(current_params)
  res = []
  for x_current, x_pretrained in zip(current_params_leaves, pretrained_params_leaves):
    if x_current.shape != x_pretrained.shape:
      assert x_current.shape[-1] == x_pretrained.shape[-1]
      if x_current.shape[0] == 1 and x_current.shape[1] == 1:
        # (1, 1, self.kernel ** 2, self.k * self.heads)
        new_k2 = x_current.shape[2]
        old_k2 = x_pretrained.shape[2]
        d = x_current.shape[3]
        new_k = int(jnp.sqrt(new_k2))
        old_k = int(jnp.sqrt(old_k2))
        x_pretrained = jnp.reshape(x_pretrained, (old_k, old_k, d))
        new_param = jax.image.resize(x_pretrained, (new_k, new_k, d), 'bicubic')
        new_param = jnp.reshape(new_param, (1, 1, new_k2, d))
      else:
        a = int(jnp.sqrt(x_current.shape[0]))
        b = int(jnp.sqrt(x_pretrained.shape[0]))
        h = x_current.shape[-1]
        x_pretrained = jnp.reshape(x_pretrained, (b, b, h))
        new_param = jax.image.resize(x_pretrained, (a, a, h), 'bicubic')
        new_param = jnp.reshape(new_param, (a * a, h))
    else:
      new_param = x_pretrained
    res.append(jnp.asarray(new_param, x_current.dtype))
  new_params = tree_util.tree_unflatten(current_params_structure, res)
  return state.replace(params=new_params)
