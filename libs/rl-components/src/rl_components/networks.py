import distrax
import flax.linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, jnp.ndarray]:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Separate actor and critic paths for stability
        actor_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor_x = activation(actor_x)
        actor_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor_x)
        actor_x = activation(actor_x)
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor_x)
        probs = distrax.Categorical(logits=actor_mean)

        critic_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        critic_x = activation(critic_x)
        critic_x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic_x)
        critic_x = activation(critic_x)
        critic_value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic_x)

        return probs, jnp.squeeze(critic_value, axis=-1)
