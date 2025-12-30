import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("generate_data_2025-12-30_11-56.pkl", "rb") as f:
    data = pickle.load(f)

ep = 0
skip_first = 30
P_loads, elec_price = (
    data["P_loads"][ep, skip_first:],
    data["elec_price"][ep, skip_first:],
)
u, y, T_s_min, T_r_min, q_r_min = (
    data["u"][ep, skip_first:],
    data["y"][ep, skip_first:-1],
    data["T_s_min"][ep, skip_first:],
    data["T_r_min"][ep, skip_first:],
    data["q_r_min"][ep, skip_first:],
)

# economic cost
economic_cost = data["economic_cost"][ep, skip_first:]

# efficiency
eff = -np.sum(P_loads, axis=1, keepdims=True) / y[:, [20]]

# constraint violations
violation_cost = data["constraint_violation_cost"][ep, skip_first:]

# disturbance sum
disturbance_sum = -np.sum(P_loads, axis=1, keepdims=True)

# create features
window_length = 288
num_points = 500
feature_vector = np.hstack((eff, economic_cost, violation_cost, disturbance_sum)).T

np.random.seed(1)
starts = np.random.randint(
    0, feature_vector.shape[1] - window_length + 1, size=num_points
)
ends = starts + window_length - 1
# cumulative sums along axis=1
c1 = np.cumsum(feature_vector, axis=1)
c2 = np.cumsum(feature_vector**2, axis=1)

# shift for window sums
sum1 = c1[:, ends] - np.where(starts > 0, c1[:, starts - 1], 0)
sum2 = c2[:, ends] - np.where(starts > 0, c2[:, starts - 1], 0)

# moving mean and variance
mean = sum1 / window_length
var = (sum2 / window_length) - mean**2

X = np.vstack((mean, var)).T

with open("monitoring_data_set.pkl", "wb") as f:
    pickle.dump(
        X,
        f,
    )


fig, ax = plt.subplots(4, 1, sharex=True)
fig.suptitle("Raw features")
ax[0].plot(eff, "o")
ax[0].set_ylabel("Efficiency")
ax[1].plot(economic_cost, "o")
ax[1].set_ylabel("Economic cost")
ax[2].plot(violation_cost, "o")
ax[2].set_ylabel("Constraint violations")
ax[3].plot(disturbance_sum, "o")
ax[3].set_ylabel("Disturbance sum")

fig, ax = plt.subplots(3, 2, sharex=False)
fig.suptitle("Features")
ax[0, 0].plot(X[:, 3], X[:, 0], "o")
ax[1, 0].plot(X[:, 3], X[:, 1], "o")
ax[2, 0].plot(X[:, 3], X[:, 2], "o")
ax[0, 1].plot(X[:, 7], X[:, 4], "o")
ax[1, 1].plot(X[:, 7], X[:, 5], "o")
ax[2, 1].plot(X[:, 7], X[:, 6], "o")
plt.show()
