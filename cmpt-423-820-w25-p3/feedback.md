Team: MatCrunch
Overall grade: 92%

# Correct implementation
Grade: 100%
Grade weight: 60%

# Distortion plot
Grade: 100%
Grade weight: 10%

# Cluster centroid figures
Grade: 100%
Grade weight: 10%

# Discussion of results and method
Grade: 60%
Grade weight: 20%

"so even if the distortion is reduced with a higher k, it doesn’t necessarily improve digit separation." -> With K-Means, you can actually improve digit separation by increasing k and placing distinct digits into distinct centroids. However, this does not necessarily mean that images showing the same digit are merged into the same cluster. The report also should have discussed different methods for moving away from using simple euclidean distances, for example with deep learning methods or a custom kernel function.
