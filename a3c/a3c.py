
import os


class A3C(object):

    def __init__(self, is_master,
                 acnet_global, acnet_local, state_to_input,
                 policy, rollout,
                 discount, train_steps, step_counter,
                 interval_save):
        self.is_master = is_master
        self.acnet_global = acnet_global
        self.acnet_local = acnet_local
        self.state_to_input = state_to_input
        self.policy = policy
        self.rollout = rollout
        self.discount = discount
        self.train_steps = train_steps
        self.step_counter = step_counter
        self.interval_save = interval_save

    def set_output(self, output):
        self.output = output

    def train(self, env):
        rollout = self.rollout
        acnet_local = self.acnet_local
        acnet_global = self.acnet_global
        step_counter = self.step_counter
        step = step_counter.step_count()
        if self.is_master:
            last_step = step
            self.save_weights(step)

        state = env.reset()
        state = self.state_to_input(state)
        while step <= self.train_steps:
            acnet_local.sync()
            rollout.reset(state)
            for t in range(rollout.maxlen):
                action_values = acnet_local.action_values([state])[0]
                action = self.policy.select_action(action_values)
                state, reward, done, info = env.step(action)
                state = self.state_to_input(state)
                rollout.append(state, action, reward, done)
                if done:
                    state = env.reset()
                    state = self.state_to_input(state)
                    break
            b_state_1more = rollout.get_batch_state()
            b_value = acnet_local.state_value(b_state_1more)
            b_action_1h, b_adv, b_target = rollout.get_batch_target(b_value)
            b_state = b_state_1more[:-1]
            acnet_local.train_on_batch(b_state, b_action_1h, b_adv, b_target)
            step_counter.increment(t)
            step = step_counter.step_count()
            if self.is_master:
                if step - last_step > self.interval_save:
                    self.save_weights(step)
                    last_step = step
                print('training step {}/{}'.format(step, self.train_steps))

    def save_weights(self, step):
        weights_save = os.path.join(self.output, 'weights_{}.p'.format(step))
        self.acnet_global.save_weights(weights_save)
        print('global net weights written to {}'.format(weights_save))

