import kagglehub

# Download latest version
# Anti-UAV410
path_1 = kagglehub.dataset_download("farhanzzzzz/anti-uav410")
path_2 = kagglehub.dataset_download("muhammadsaoodsarwar/drone-vs-bird")

print("Path to dataset files:", path_2)