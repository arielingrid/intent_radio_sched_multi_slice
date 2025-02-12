from collections import deque
from typing import Optional, Union

import numpy as np
import os
from tqdm import tqdm
from agents import util
from gymnasium import spaces
#from stable_baselines3.common.callbacks import CheckpointCallback

from agents.common import (
    calculate_slice_ue_obs,
    get_metric_value,
    intent_drift_calc,
)

from agents.rl_simple import PIAgent
#from agents.train_agent import PIAgentTrain
#from agents.sb3_callbacks import CustomEvalCallback as EvalCallback
#from agents.sb3_sched import IBSchedSB3
from sixg_radio_mgmt import Agent, MARLCommEnv


class PIAgentSched(Agent):
    def __init__(
        self,
        env: MARLCommEnv,
        max_number_ues: int,
        max_number_slices: int,
        max_number_basestations: int,
        num_available_rbs: np.ndarray,
        eval_env: Optional[MARLCommEnv] = None,
        seed: int = np.random.randint(1000),
        agent_name: str = "pi_agent",
        episode_evaluation_freq: Optional[int] = None,
        number_evaluation_episodes: Optional[int] = None,
        checkpoint_episode_freq: Optional[int] = None,
        eval_initial_env_episode: Optional[int] = None,
    ) -> None:
        super().__init__(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
            seed=seed,
        )
        
        self.seed = 10
        
        self.fake_agent = PIAgent(
            env,
            max_number_ues,
            max_number_slices,
            max_number_basestations,
            num_available_rbs,
            enable_sort_slices=False,
        )
        self.agent = None

    def init_agent(self) -> None:
        self.env.set_agent_functions(
        self.fake_agent.obs_space_format,
        self.fake_agent.action_format,
        self.fake_agent.calculate_reward,
        self.fake_agent.get_obs_space(),
        self.fake_agent.get_action_space(),
    )
        self.fake_agent.init_agent()
        
    def train(self, total_timesteps: int) -> None:
        self.fake_agent.train(total_timesteps)
    
    def load(
        self, agent_name, scenario, method="last", finetune=False
    ) -> None:
        log_dir="/home/lasse_rtx/Repos2/intent_radio_sched_multi_slice/output"
        model_filename="/home/lasse_rtx/Repos2/intent_radio_sched_multi_slice/output/best.npz" 
        logger = util.create_logger(name='pi_agent_test_1', log_dir=log_dir)
        task = util.create_task(logger=logger)
        task.seed(self.seed)
        self.solution = util.create_solution(device='cpu:0')
        model_file = os.path.join(log_dir, model_filename)
        self.solution.load(model_file)
    
    def step(self, obs_space: Optional[Union[np.ndarray, dict]]) -> np.ndarray:   
        action = self.solution.get_action(obs_space)
        return action
    
    def obs_space_format(self, obs_space: dict) -> Union[np.ndarray, dict]:
        # For each slice keep  nine metrics defined
        # for each slice and UE: spectral efficiency, served throughput,
        # effective throughput, buffer occupancy, packet loss rate, re-
        # quested throughput, average buffer latency, long-term served
        # throughput and the fifth-percentile served throughput.
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        self.fake_agent.obs_space_format(
            obs_space
        )  # updating internal fake_agent variables
        self.fake_agent.last_unformatted_obs.appendleft(obs_space)

        # Inter-slice observation
        formatted_obs_space = {}
        formatted_obs_space["player_0"] = {
            "observations": np.array([]),
            "action_mask": self.fake_agent.last_unformatted_obs[0][
                "basestation_slice_assoc"
            ][0].astype(np.int8),
        }

        ordered_metrics = [
            "requirements",
            "spectral_efficiencies",
            "pkt_throughputs",
            "pkt_effective_thrs",
            "buffer_occupancies",
            "buffer_latencies",
            "pkt_loss_rates",
            "requested_thrs",
        ]
        dict_metrics = {
            "requirements": np.array([]),
            "spectral_efficiencies": np.array([]),
            "pkt_throughputs": np.array([]),
            "pkt_effective_thrs": np.array([]),
            "buffer_occupancies": np.array([]),
            "buffer_latencies": np.array([]),
            "pkt_loss_rates": np.array([]),
            "requested_thrs": np.array([]),
        }

        # intra-slice observations
        for agent_idx in range(1, obs_space["slice_ue_assoc"].shape[0] + 1):
            slice_ues = self.fake_agent.last_unformatted_obs[0][
                "slice_ue_assoc"
            ][agent_idx - 1].nonzero()[0]

            requirements = np.zeros(3)
            if slice_ues.shape[0] != 0:
                for parameter in self.fake_agent.last_unformatted_obs[0][
                    "slice_req"
                ][f"slice_{agent_idx-1}"]["parameters"].values():
                    if parameter["name"] == "reliability":
                        requirements[0] = parameter["value"]
                    elif parameter["name"] == "latency":
                        requirements[1] = parameter["value"]
                    elif parameter["name"] == "throughput":
                        requirements[2] = parameter["value"]
            dict_metrics["requirements"] = np.append(
                dict_metrics["requirements"], requirements
            )
            # Pkt size
            pkt_size = (
                self.fake_agent.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["ues"]["message_size"]
                if slice_ues.shape[0] != 0
                else 0
            )

            # Spectral efficiency
            if slice_ues.shape[0] != 0:
                spectral_eff = np.mean(
                    self.fake_agent.last_unformatted_obs[0][
                        "spectral_efficiencies"
                    ][0, slice_ues, :],
                    axis=1,
                )
            else:
                spectral_eff = np.array([0])
            dict_metrics["spectral_efficiencies"] = np.append(
                dict_metrics["spectral_efficiencies"], np.mean(spectral_eff)
            )

            # Served Throughput
            served_thr = (
                self.fake_agent.last_unformatted_obs[0]["pkt_throughputs"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_throughputs"] = np.append(
                dict_metrics["pkt_throughputs"],
                np.mean(self.pkts_to_mbps(served_thr, pkt_size)),
            )

            # Effective Throughput
            eff_thr = (
                self.fake_agent.last_unformatted_obs[0]["pkt_effective_thr"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_effective_thrs"] = np.append(
                dict_metrics["pkt_effective_thrs"],
                np.mean(self.pkts_to_mbps(eff_thr, pkt_size)),
            )

            # Buffer Occ.
            buffer_occ = (
                self.fake_agent.last_unformatted_obs[0]["buffer_occupancies"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["buffer_occupancies"] = np.append(
                dict_metrics["buffer_occupancies"], np.mean(buffer_occ)
            )

            # Buffer latencies
            buffer_latencies = (
                self.fake_agent.last_unformatted_obs[0]["buffer_latencies"][
                    slice_ues
                ]
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["buffer_latencies"] = np.append(
                dict_metrics["buffer_latencies"], np.mean(buffer_latencies)
            )

            # Packet loss rate
            pkt_loss_rate = (
                get_metric_value(
                    "reliability",
                    self.fake_agent.last_unformatted_obs,
                    agent_idx - 1,
                    slice_ues,
                    reliability_pkt_loss=True,
                )
                if slice_ues.shape[0] != 0
                else np.array([0])
            )
            dict_metrics["pkt_loss_rates"] = np.append(
                dict_metrics["pkt_loss_rates"], np.mean(pkt_loss_rate)
            )

            # Requested throughput
            slice_traffic_req = (
                self.fake_agent.last_unformatted_obs[0]["slice_req"][
                    f"slice_{agent_idx-1}"
                ]["ues"]["traffic"]
                if np.isclose(
                    self.fake_agent.last_unformatted_obs[0][
                        "basestation_slice_assoc"
                    ][0, agent_idx - 1],
                    1,
                )
                else 0
            )
            dict_metrics["requested_thrs"] = np.append(
                dict_metrics["requested_thrs"],
                slice_traffic_req,
            )
        for key in ordered_metrics:
            formatted_obs_space["player_0"]["observations"] = np.concatenate(
                (
                    formatted_obs_space["player_0"]["observations"],
                    dict_metrics[key],
                )
            )

        self.last_formatted_obs = formatted_obs_space
        
        return formatted_obs_space["player_0"]["observations"]
    
    def calculate_reward(self, obs_space: Union[np.ndarray, dict]) -> float:
        assert isinstance(
            self.env, MARLCommEnv
        ), "Environment must be MARLCommEnv"
        intent_drift = intent_drift_calc(
            self.fake_agent.last_unformatted_obs,
            self.fake_agent.max_number_ues_slice,
            self.fake_agent.intent_overfulfillment_rate,
        )
        valid_intents = np.array([])
        weights = np.array([])
        reward = {}
        for player_idx in np.arange(
            1, self.env.comm_env.max_number_slices + 1
        ):
            slice_ues = self.fake_agent.last_unformatted_obs[0][
                "slice_ue_assoc"
            ][player_idx - 1].nonzero()[0]
            if slice_ues.shape[0] == 0:
                continue
            (
                _,
                intent_drift_slice,
            ) = calculate_slice_ue_obs(
                self.fake_agent.max_number_ues_slice,
                intent_drift,
                player_idx - 1,
                slice_ues,
                self.fake_agent.last_unformatted_obs[0]["slice_req"],
            )
            valid_intent = intent_drift_slice[
                np.logical_not(np.isclose(intent_drift_slice, -2))
            ]
            valid_intents = np.append(
                valid_intents,
                valid_intent,
            )
            weight_value = (
                2
                if bool(
                    self.fake_agent.last_unformatted_obs[0]["slice_req"][
                        f"slice_{player_idx - 1}"
                    ]["priority"]
                )
                else 1
            )
            weights = np.append(
                weights, weight_value * np.ones_like(valid_intent)
            )
        valid_intents[
            valid_intents > 0
        ] = 0  # It does not consider positive values
        idx_negative_intents = valid_intents < 0
        negative_intents = valid_intents[idx_negative_intents]
        negative_intents_weights = weights[idx_negative_intents]
        reward = (
            np.sum(
                negative_intents
                * negative_intents_weights
                / np.sum(negative_intents_weights)
            )
            if not np.isclose(np.sum(negative_intents_weights), 0)
            else 0
        )

        return reward

    def action_format(self, action_ori: Union[np.ndarray, dict]) -> np.ndarray:
        action = {
            "player_0": action_ori,
        }
        allocation_rbs = self.fake_agent.action_format(
            action, fixed_intra="rr"
        )

        return allocation_rbs
    
    def get_action_space(self) -> spaces.Space:
        action_space = self.fake_agent.get_action_space()

        return action_space["player_0"]
    
    def get_obs_space(self) -> spaces.Space:
        obs_space = self.fake_agent.get_obs_space()["player_0"]["observations"]  # type: ignore

        return obs_space
    
    def pkts_to_mbps(self, pkts: np.ndarray, pkt_size: float) -> np.ndarray:
        return pkts * pkt_size / 1e6
    
    @staticmethod
    def sb3_load_path(agent_name, scenario, method="last"):
        if method == "last":
            return f"./agents/models/{scenario}/final_{agent_name}.zip"
        elif method == "best":
            return (
                f"./agents/models/{scenario}/best_{agent_name}/best_model.zip"
            )
        elif isinstance(method, int):
            return f"./agents/models/{scenario}/{agent_name}/{agent_name}_{int(method*1000)}_steps.zip"
        else:
            raise ValueError(f"Invalid method {method} for finetune load")

#if __name__ == '__main__':
#    self.fake_agent.init_agent()
