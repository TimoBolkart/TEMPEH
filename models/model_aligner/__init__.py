from .base_model import BaseModel

# feature net
from .feature_net.feature_net_2d import Model as FeatureNet2D

# sparse point net
from .sparse_point_net.global_sparse_point_net import Model as GlobalSparsePointNet
from .sparse_point_net.global_sparse_point_net_transformer import Model as GlobalSparsePointNetTransformer

# densify net
from .densify_net.local_densify_net import Model as LocalDensifyNet
