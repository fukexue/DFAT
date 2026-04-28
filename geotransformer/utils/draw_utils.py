"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 22.02.2021
"""
import os,torch,sys,copy
import numpy as np
import open3d as o3d
from geotransformer.utils.pointcloud import get_nearest_neighbor

cwd = os.getcwd()
sys.path.append(cwd)

def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def get_red():
    """
    Get color blue for rendering
    """
    return [1, 0, 0]

def to_tensor(array):
    """
    Convert array to tensor
    """
    if (not isinstance(array, torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array


def to_array(tensor):
    """
    Conver tensor to array
    """
    if (not isinstance(tensor, np.ndarray)):
        if (tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def get_inlier(src_pcd, tgt_pcd, gt, inlier_distance_threshold = 0.1):
    """
    Compute inlier with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    rot, trans = to_tensor(gt[:3,:3]), to_tensor(gt[:3,3][:,None])

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)

    dist = torch.norm(src_pcd- tgt_pcd,dim=1)

    c_inlier_where = np.where(to_array(dist < inlier_distance_threshold))

    return tuple(c_inlier_where[0])

def get_outlier(src_pcd, tgt_pcd, gt, inlier_distance_threshold = 0.1):
    """
    Compute inlier with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    rot, trans = to_tensor(gt[:3,:3]), to_tensor(gt[:3,3][:,None])

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)

    dist = torch.norm(src_pcd- tgt_pcd,dim=1)

    c_inlier_where = np.where(to_array(dist > inlier_distance_threshold))

    return tuple(c_inlier_where[0])


def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent


def draw_single_attention_v1(points, nodes, nodes_keys, nodes_val):
    ########################################
    # 1. input point cloud
    points = to_array(points)
    nodes = to_array(nodes)
    nodes_keys = to_array(nodes_keys)
    nodes_val = to_array(nodes_val)
    nodes_val_norm = (nodes_val - nodes_val.min()) / (nodes_val.max() - nodes_val.min())

    dist, index = get_nearest_neighbor(points, nodes, True)
    points_color = lighter(get_blue(), 1 - nodes_val_norm[index])

    points_pcd = to_o3d_pcd(points)
    points_pcd.colors = o3d.utility.Vector3dVector(points_color)
    points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    # nodes_pcd = to_o3d_pcd(nodes)
    # nodes_pcd.paint_uniform_color(get_yellow())
    # nodes_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    # nodes_keypoints = nodes[nodes_keys, :]
    # nodes_keypoints_pcd = to_o3d_pcd(nodes_keypoints)
    # nodes_keypoints_pcd.paint_uniform_color(get_red())
    # nodes_keypoints_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(points_pcd)
    # vis1.add_geometry(nodes_pcd)
    # vis1.add_geometry(nodes_keypoints_pcd)

    while True:
        vis1.update_geometry(points_pcd)
        # vis1.update_geometry(nodes_pcd)
        # vis1.update_geometry(nodes_keypoints_pcd)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_single_attention_v2(points, nodes, nodes_keys, nodes_val, windows_name='win'):
    ########################################
    # 1. input point cloud
    points = to_array(points)
    nodes = to_array(nodes)
    nodes_keys = to_array(nodes_keys)
    nodes_val = to_array(nodes_val)
    nodes_val_trans = np.repeat(nodes_val.mean(0).transpose(1, 0), 3, axis=1)
    nodes_val_norm = (nodes_val_trans - nodes_val_trans.min() + 0.2) / (nodes_val_trans.max() - nodes_val_trans.min() + 0.2)

    dist, index = get_nearest_neighbor(points, nodes, True)
    node_color_tmp = np.ones_like(nodes)*0.05
    node_color_tmp[nodes_keys, :] = nodes_val_norm
    points_color = lighter(get_blue(), 1 - node_color_tmp[index])

    points_pcd = to_o3d_pcd(points)
    points_pcd.colors = o3d.utility.Vector3dVector(points_color)
    points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name=windows_name, width=640, height=520, left=0, top=0)
    vis1.add_geometry(points_pcd)

    while True:
        vis1.update_geometry(points_pcd)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_single_attention_v2_2(points0, nodes0, nodes_keys0, nodes_val0, points1, nodes1, nodes_keys1, nodes_val1, windows_name='win'):
    ########################################
    # 1. input point cloud
    def create_pcd(points, nodes, nodes_keys, nodes_val):
        points = to_array(points)
        nodes = to_array(nodes)
        nodes_keys = to_array(nodes_keys)
        nodes_val = to_array(nodes_val)
        nodes_val_trans = np.repeat(nodes_val.mean(0).transpose(1, 0), 3, axis=1)
        nodes_val_norm = (nodes_val_trans - nodes_val_trans.min() + 0.2) / (nodes_val_trans.max() - nodes_val_trans.min() + 0.2)
        dist, index = get_nearest_neighbor(points, nodes, True)
        node_color_tmp = np.ones_like(nodes)*0.05
        node_color_tmp[nodes_keys, :] = nodes_val_norm
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    points_pcd_0 = create_pcd(points0, nodes0, nodes_keys0, nodes_val0)
    points_pcd_1 = create_pcd(points1, nodes1, nodes_keys1, nodes_val1)

    ########################################
    vis0 = o3d.visualization.Visualizer()
    vis0.create_window(window_name=windows_name+'_src_points', width=640, height=520, left=0, top=0)
    vis0.add_geometry(points_pcd_0)
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name=windows_name+'_ref_points', width=640, height=520, left=640, top=0)
    vis1.add_geometry(points_pcd_1)

    while True:
        vis0.update_geometry(points_pcd_0)
        if not vis0.poll_events():
            break
        vis0.update_renderer()
        vis1.update_geometry(points_pcd_1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis0.destroy_window()
    vis1.destroy_window()


def draw_single_attention_v2_2(points0, nodes0, nodes_keys0, nodes_val0, points1, nodes1, nodes_keys1, nodes_val1, windows_name='win'):
    ########################################
    # 1. input point cloud
    def create_pcd(points, nodes, nodes_keys, nodes_val):
        points = to_array(points)
        nodes = to_array(nodes)
        nodes_keys = to_array(nodes_keys)
        nodes_val = to_array(nodes_val)
        nodes_val_trans = np.repeat(nodes_val.mean(0).transpose(1, 0), 3, axis=1)
        nodes_val_norm = (nodes_val_trans - nodes_val_trans.min() + 0.2) / (nodes_val_trans.max() - nodes_val_trans.min() + 0.2)
        dist, index = get_nearest_neighbor(points, nodes, True)
        node_color_tmp = np.ones_like(nodes)*0.05
        node_color_tmp[nodes_keys, :] = nodes_val_norm
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    points_pcd_0 = create_pcd(points0, nodes0, nodes_keys0, nodes_val0)
    points_pcd_1 = create_pcd(points1, nodes1, nodes_keys1, nodes_val1)

    ########################################
    vis0 = o3d.visualization.Visualizer()
    vis0.create_window(window_name=windows_name+'_src_points', width=640, height=520, left=0, top=0)
    vis0.add_geometry(points_pcd_0)
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name=windows_name+'_ref_points', width=640, height=520, left=640, top=0)
    vis1.add_geometry(points_pcd_1)

    while True:
        vis0.update_geometry(points_pcd_0)
        if not vis0.poll_events():
            break
        vis0.update_renderer()
        vis1.update_geometry(points_pcd_1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis0.destroy_window()
    vis1.destroy_window()



##################################################################

def draw_twopc_cp(points0, nodes0, points1, nodes1, points2, nodes2, points3, nodes3, gt_tsfm=None, cp_our=None, cp_geotran=None, key_src=None, key_tgt=None):
    ########################################
    # 1. input point cloud
    def create_our_pcd(points, nodes):
        points = to_array(points)
        nodes = to_array(nodes)
        dist, index = get_nearest_neighbor(points, nodes, True)
        node_color_tmp = np.ones_like(nodes)*0.9
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    def create_geotrans_pcd(points, nodes):
        points = to_array(points)
        nodes = to_array(nodes)
        dist, index = get_nearest_neighbor(points, nodes, True)
        node_color_tmp = np.ones_like(nodes)*0.9
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    def create_correspondence(src_raw, tgt_raw, cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold=0.2):
        translate = [1.6, -1.0, 0]
        # translate = [0, 0, 0]

        src_pcd = to_o3d_pcd(src_raw)
        tgt_pcd = to_o3d_pcd(tgt_raw)
        src_pcd.paint_uniform_color(get_yellow())
        tgt_pcd.paint_uniform_color(get_blue())
        src_pcd.transform(gt_tsfm)
        tgt_pcd.translate(translate)

        gt_inliers = get_inlier(cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold)
        cp_temp = np.arange(cp_src.shape[0])[:, None].repeat(2, axis=1) + np.array([0, cp_src.shape[0]])
        colors = [[0, 1, 0] if i in gt_inliers else [1, 0, 0] for i in range(cp_src.shape[0])]
        line_set = o3d.geometry.LineSet()
        src_pcd_cp = to_o3d_pcd(cp_src)
        tgt_pcd_cp = to_o3d_pcd(cp_tgt)
        src_pcd_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_cp.points, tgt_pcd_cp.points]))
        line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return src_pcd, tgt_pcd, line_set

    def create_pcd(points):
        points = to_array(points)
        points_pcd = to_o3d_pcd(points)
        points_pcd.paint_uniform_color(get_red())
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    points_pcd_0 = create_our_pcd(points0, nodes0)
    points_pcd_1 = create_our_pcd(points1, nodes1)
    points_pcd_2 = create_geotrans_pcd(points2, nodes2)
    points_pcd_3 = create_geotrans_pcd(points3, nodes3)
    if cp_our is not None and cp_geotran is not None:
        points_pcd_0_our, points_pcd_1_our, lineset_pcd_our = create_correspondence(points0, points1, cp_our[0], cp_our[1], gt_tsfm)
        points_pcd_2_geo, points_pcd_3_geo, lineset_pcd_geo = create_correspondence(points2, points3, cp_geotran[0], cp_geotran[1], gt_tsfm)

    if key_src is not None and key_tgt is not None:
        points_pcd_0_key = create_pcd(nodes0[key_src][None, :])
        points_pcd_1_key = create_pcd(nodes1[key_tgt][None, :])
        points_pcd_2_key = create_pcd(nodes2[key_src][None, :])
        points_pcd_3_key = create_pcd(nodes3[key_tgt][None, :])

    ########################################
    vis0 = o3d.visualization.Visualizer()
    vis0.create_window(window_name='our_src_points', width=640, height=520, left=0, top=0)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis1.json')
    ctr0 = vis0.get_view_control()
    vis0.add_geometry(points_pcd_0)
    if key_src is not None:
        vis0.add_geometry(points_pcd_0_key)
    ctr0.convert_from_pinhole_camera_parameters(param)
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='our_ref_points', width=640, height=520, left=715, top=0)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis2.json')
    ctr1 = vis1.get_view_control()
    vis1.add_geometry(points_pcd_1)
    if key_tgt is not None:
        vis1.add_geometry(points_pcd_1_key)
    ctr1.convert_from_pinhole_camera_parameters(param)
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='geotrans_src_points', width=640, height=520, left=0, top=620)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis1.json')
    ctr2 = vis2.get_view_control()
    vis2.add_geometry(points_pcd_2)
    if key_src is not None:
        vis2.add_geometry(points_pcd_2_key)
    ctr2.convert_from_pinhole_camera_parameters(param)
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='geotrans_ref_points', width=640, height=520, left=715, top=620)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis2.json')
    ctr3 = vis3.get_view_control()
    vis3.add_geometry(points_pcd_3)
    if key_tgt is not None:
        vis3.add_geometry(points_pcd_3_key)
    ctr3.convert_from_pinhole_camera_parameters(param)
    if cp_our is not None and cp_geotran is not None:
        vis4 = o3d.visualization.Visualizer()
        vis4.create_window(window_name='our_cp', width=640, height=520, left=1360, top=0)
        vis4.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_3DLoMatch_id138.json')
        param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/covis_vis4.json')
        ctr4 = vis4.get_view_control()
        vis4.add_geometry(points_pcd_0_our)
        vis4.add_geometry(points_pcd_1_our)
        vis4.add_geometry(lineset_pcd_our)
        ctr4.convert_from_pinhole_camera_parameters(param)
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name='geotrans_cp', width=640, height=520, left=1360, top=620)
        vis5.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/render_json/RenderOption_3DLoMatch_id138.json')
        ctr5 = vis5.get_view_control()
        vis5.add_geometry(points_pcd_2_geo)
        vis5.add_geometry(points_pcd_3_geo)
        vis5.add_geometry(lineset_pcd_geo)
        ctr5.convert_from_pinhole_camera_parameters(param)

    while True:
        vis0.update_geometry(points_pcd_0)
        if not vis0.poll_events():
            break
        vis0.update_renderer()
        vis1.update_geometry(points_pcd_1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
        vis2.update_geometry(points_pcd_2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
        # param = vis2.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('vis_tmp1.json', param)
        vis3.update_geometry(points_pcd_3)
        if not vis3.poll_events():
            break
        vis3.update_renderer()
        # param = vis3.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('vis_tmp2.json', param)

        if cp_our is not None and cp_geotran is not None:
            vis4.update_geometry(points_pcd_0_our)
            vis4.update_geometry(points_pcd_1_our)
            vis4.update_geometry(lineset_pcd_our)
            if not vis4.poll_events():
                break
            vis4.update_renderer()
            # param = vis4.get_view_control().convert_to_pinhole_camera_parameters()
            # o3d.io.write_pinhole_camera_parameters('view1.json', param)

            vis5.update_geometry(points_pcd_2_geo)
            vis5.update_geometry(points_pcd_3_geo)
            vis5.update_geometry(lineset_pcd_geo)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

    vis0.destroy_window()
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()
##################################################################



def draw_single_attention_v2_v1(points0, nodes0, nodes_keys0, nodes_val0, points1, nodes1, nodes_keys1, nodes_val1, points2, nodes2, nodes_val2, points3, nodes3, nodes_val3, gt_tsfm=None, cp_our=None, cp_geotran=None, key_src=None, key_tgt=None):
    ########################################
    # 1. input point cloud
    def create_our_pcd(points, nodes, nodes_keys, nodes_val):
        points = to_array(points)
        nodes = to_array(nodes)
        nodes_keys = to_array(nodes_keys)
        nodes_val = to_array(nodes_val)
        nodes_val_trans = np.repeat(nodes_val.mean(0).transpose(1, 0), 3, axis=1)
        nodes_val_norm = (nodes_val_trans - nodes_val_trans.min() + 0.2) / (nodes_val_trans.max() - nodes_val_trans.min() + 0.2)
        dist, index = get_nearest_neighbor(points, nodes, True)
        node_color_tmp = np.ones_like(nodes)*0.05
        node_color_tmp[nodes_keys, :] = nodes_val_norm
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    def create_geotrans_pcd(points, nodes, nodes_val):
        points = to_array(points)
        nodes = to_array(nodes)
        nodes_val = to_array(nodes_val)
        nodes_val_norm = (nodes_val - nodes_val.min()) / (nodes_val.max() - nodes_val.min()) * (1-nodes_val.min()*10000) + nodes_val.min()*10000
        node_color_tmp = np.ones_like(nodes) * nodes_val_norm.reshape(-1, 1)
        dist, index = get_nearest_neighbor(points, nodes, True)
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    def create_correspondence(src_raw, tgt_raw, cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold=0.2):
        translate = [-1.3, -1.5, 0]
        # translate = [0, 0, 0]

        src_pcd = to_o3d_pcd(src_raw)
        tgt_pcd = to_o3d_pcd(tgt_raw)
        src_pcd.paint_uniform_color(get_yellow())
        tgt_pcd.paint_uniform_color(get_blue())
        src_pcd.transform(gt_tsfm)
        tgt_pcd.translate(translate)

        gt_inliers = get_inlier(cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold)
        cp_temp = np.arange(cp_src.shape[0])[:, None].repeat(2, axis=1) + np.array([0, cp_src.shape[0]])
        colors = [[0, 1, 0] if i in gt_inliers else [1, 0, 0] for i in range(cp_src.shape[0])]
        line_set = o3d.geometry.LineSet()
        src_pcd_cp = to_o3d_pcd(cp_src)
        tgt_pcd_cp = to_o3d_pcd(cp_tgt)
        src_pcd_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_cp.points, tgt_pcd_cp.points]))
        line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return src_pcd, tgt_pcd, line_set

    def create_pcd(points):
        points = to_array(points)
        points_pcd = to_o3d_pcd(points)
        points_pcd.paint_uniform_color(get_red())
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    points_pcd_0 = create_our_pcd(points0, nodes0, nodes_keys0, nodes_val0)
    points_pcd_1 = create_our_pcd(points1, nodes1, nodes_keys1, nodes_val1)
    points_pcd_2 = create_geotrans_pcd(points2, nodes2, nodes_val2)
    points_pcd_3 = create_geotrans_pcd(points3, nodes3, nodes_val3)
    if cp_our is not None and cp_geotran is not None:
        points_pcd_0_our, points_pcd_1_our, lineset_pcd_our = create_correspondence(points0, points1, cp_our[0], cp_our[1], gt_tsfm)
        points_pcd_2_geo, points_pcd_3_geo, lineset_pcd_geo = create_correspondence(points2, points3, cp_geotran[0], cp_geotran[1], gt_tsfm)

    if key_src is not None and key_tgt is not None:
        points_pcd_0_key = create_pcd(nodes0[key_src][None, :])
        points_pcd_1_key = create_pcd(nodes1[key_tgt][None, :])
        points_pcd_2_key = create_pcd(nodes2[key_src][None, :])
        points_pcd_3_key = create_pcd(nodes3[key_tgt][None, :])

    ########################################
    vis0 = o3d.visualization.Visualizer()
    vis0.create_window(window_name='our_src_points', width=640, height=520, left=0, top=0)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis1.json')
    ctr0 = vis0.get_view_control()
    vis0.add_geometry(points_pcd_0)
    if key_src is not None:
        vis0.add_geometry(points_pcd_0_key)
    ctr0.convert_from_pinhole_camera_parameters(param)
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='our_ref_points', width=640, height=520, left=715, top=0)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis2.json')
    ctr1 = vis1.get_view_control()
    vis1.add_geometry(points_pcd_1)
    if key_tgt is not None:
        vis1.add_geometry(points_pcd_1_key)
    ctr1.convert_from_pinhole_camera_parameters(param)
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='geotrans_src_points', width=640, height=520, left=0, top=620)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis1.json')
    ctr2 = vis2.get_view_control()
    vis2.add_geometry(points_pcd_2)
    if key_src is not None:
        vis2.add_geometry(points_pcd_2_key)
    ctr2.convert_from_pinhole_camera_parameters(param)
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='geotrans_ref_points', width=640, height=520, left=715, top=620)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis2.json')
    ctr3 = vis3.get_view_control()
    vis3.add_geometry(points_pcd_3)
    if key_tgt is not None:
        vis3.add_geometry(points_pcd_3_key)
    ctr3.convert_from_pinhole_camera_parameters(param)
    if cp_our is not None and cp_geotran is not None:
        vis4 = o3d.visualization.Visualizer()
        vis4.create_window(window_name='our_cp', width=640, height=520, left=1360, top=0)
        vis4.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_vis4.json')
        param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis4.json')
        ctr4 = vis4.get_view_control()
        vis4.add_geometry(points_pcd_0_our)
        vis4.add_geometry(points_pcd_1_our)
        vis4.add_geometry(lineset_pcd_our)
        ctr4.convert_from_pinhole_camera_parameters(param)
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name='geotrans_cp', width=640, height=520, left=1360, top=620)
        vis5.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/render_json/RenderOption_vis4.json')
        ctr5 = vis5.get_view_control()
        vis5.add_geometry(points_pcd_2_geo)
        vis5.add_geometry(points_pcd_3_geo)
        vis5.add_geometry(lineset_pcd_geo)
        ctr5.convert_from_pinhole_camera_parameters(param)

    while True:
        vis0.update_geometry(points_pcd_0)
        if not vis0.poll_events():
            break
        vis0.update_renderer()
        vis1.update_geometry(points_pcd_1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
        vis2.update_geometry(points_pcd_2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
        # param = vis2.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('vis_tmp1.json', param)
        vis3.update_geometry(points_pcd_3)
        if not vis3.poll_events():
            break
        vis3.update_renderer()
        # param = vis3.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('vis_tmp2.json', param)

        if cp_our is not None and cp_geotran is not None:
            vis4.update_geometry(points_pcd_0_our)
            vis4.update_geometry(points_pcd_1_our)
            vis4.update_geometry(lineset_pcd_our)
            if not vis4.poll_events():
                break
            vis4.update_renderer()
            # param = vis4.get_view_control().convert_to_pinhole_camera_parameters()
            # o3d.io.write_pinhole_camera_parameters('view1.json', param)

            vis5.update_geometry(points_pcd_2_geo)
            vis5.update_geometry(points_pcd_3_geo)
            vis5.update_geometry(lineset_pcd_geo)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

    vis0.destroy_window()
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()


def draw_single_attention_v2_v1_2(points0, nodes0, nodes_keys0, nodes_val0, points1, nodes1, nodes_keys1, nodes_val1, points2, nodes2, nodes_val2, points3, nodes3, nodes_val3, gt_tsfm=None, cp_our=None, cp_geotran=None, cp_geotran_c=None, key_src=None, key_tgt=None):
    ########################################
    # 1. input point cloud
    def create_our_pcd(points, nodes, nodes_keys, nodes_val):
        points = to_array(points)
        nodes = to_array(nodes)
        nodes_keys = to_array(nodes_keys)
        nodes_val = to_array(nodes_val)
        nodes_val_trans = np.repeat(nodes_val.mean(0).transpose(1, 0), 3, axis=1)
        nodes_val_norm = (nodes_val_trans - nodes_val_trans.min() + 0.2) / (nodes_val_trans.max() - nodes_val_trans.min() + 0.2)
        dist, index = get_nearest_neighbor(points, nodes, True)
        node_color_tmp = np.ones_like(nodes)*0.05
        node_color_tmp[nodes_keys, :] = nodes_val_norm
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    def create_geotrans_pcd(points, nodes, nodes_val):
        points = to_array(points)
        nodes = to_array(nodes)
        nodes_val = to_array(nodes_val)
        nodes_val_norm = (nodes_val - nodes_val.min()) / (nodes_val.max() - nodes_val.min())*10
        node_color_tmp = np.ones_like(nodes) * nodes_val_norm.reshape(-1, 1)
        dist, index = get_nearest_neighbor(points, nodes, True)
        points_color = lighter(get_blue(), 1 - node_color_tmp[index])
        points_pcd = to_o3d_pcd(points)
        points_pcd.colors = o3d.utility.Vector3dVector(points_color)
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    def create_correspondence(src_raw, tgt_raw, cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold=0.2):
        translate = [3, 1, 0]
        # translate = [0, 0, 0]

        src_pcd = to_o3d_pcd(src_raw)
        tgt_pcd = to_o3d_pcd(tgt_raw)
        src_pcd.paint_uniform_color(get_yellow())
        tgt_pcd.paint_uniform_color(get_blue())
        src_pcd.transform(gt_tsfm)
        tgt_pcd.translate(translate)

        gt_outliers = get_outlier(cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold)
        cp_temp = np.arange(cp_src[list(gt_outliers)].shape[0])[:, None].repeat(2, axis=1) + np.array([0, cp_src[list(gt_outliers)].shape[0]])
        colors = [[1, 0, 0] for i in range(gt_outliers.__len__())]
        line_set = o3d.geometry.LineSet()
        src_pcd_cp = to_o3d_pcd(cp_src[list(gt_outliers)])
        tgt_pcd_cp = to_o3d_pcd(cp_tgt[list(gt_outliers)])
        src_pcd_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_cp.points, tgt_pcd_cp.points]))
        line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return src_pcd, tgt_pcd, line_set

    def create_correspondence_1(src_raw, tgt_raw, cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold=0.2):
        translate = [5, 0, 0]
        # translate = [0, 0, 0]

        src_pcd = to_o3d_pcd(src_raw)
        tgt_pcd = to_o3d_pcd(tgt_raw)
        src_pcd.paint_uniform_color(get_yellow())
        tgt_pcd.paint_uniform_color(get_blue())
        src_pcd.transform(gt_tsfm)
        tgt_pcd.translate(translate)

        gt_inliers = get_inlier(cp_src, cp_tgt, gt_tsfm, inlier_distance_threshold)
        cp_temp = np.arange(cp_src[list(gt_inliers)].shape[0])[:, None].repeat(2, axis=1) + np.array([0, cp_src[list(gt_inliers)].shape[0]])
        colors = [[0, 1, 0] for i in range(gt_inliers.__len__())]
        line_set = o3d.geometry.LineSet()
        src_pcd_cp = to_o3d_pcd(cp_src[list(gt_inliers)])
        tgt_pcd_cp = to_o3d_pcd(cp_tgt[list(gt_inliers)])
        src_pcd_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_cp.points, tgt_pcd_cp.points]))
        line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return src_pcd, tgt_pcd, line_set

    def create_pcd(points):
        points = to_array(points)
        points_pcd = to_o3d_pcd(points)
        points_pcd.paint_uniform_color(get_red())
        points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        return points_pcd

    points_pcd_0 = create_our_pcd(points0, nodes0, nodes_keys0, nodes_val0)
    points_pcd_1 = create_our_pcd(points1, nodes1, nodes_keys1, nodes_val1)
    points_pcd_2 = create_geotrans_pcd(points2, nodes2, nodes_val2)
    points_pcd_3 = create_geotrans_pcd(points3, nodes3, nodes_val3)
    if cp_our is not None and cp_geotran is not None:
        # points_pcd_0_our, points_pcd_1_our, lineset_pcd_our = create_correspondence(points0, points1, cp_our[0], cp_our[1], gt_tsfm)
        points_pcd_2_geo, points_pcd_3_geo, lineset_pcd_geo = create_correspondence(points2, points3, cp_geotran[0], cp_geotran[1], gt_tsfm)

    if cp_geotran_c is not None:
        _1, _2, lineset_pcd_geo_c = create_correspondence_1(points2, points3, cp_geotran_c[0], cp_geotran_c[1], gt_tsfm)

    if key_src is not None and key_tgt is not None:
        points_pcd_0_key = create_pcd(nodes0[key_src][None, :])
        points_pcd_1_key = create_pcd(nodes1[key_tgt][None, :])
        points_pcd_2_key = create_pcd(nodes2[key_src][None, :])
        points_pcd_3_key = create_pcd(nodes3[key_tgt][None, :])

    ########################################
    vis0 = o3d.visualization.Visualizer()
    vis0.create_window(window_name='our_src_points', width=640, height=520, left=0, top=0)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis1.json')
    ctr0 = vis0.get_view_control()
    vis0.add_geometry(points_pcd_0)
    if key_src is not None:
        vis0.add_geometry(points_pcd_0_key)
    ctr0.convert_from_pinhole_camera_parameters(param)
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='our_ref_points', width=640, height=520, left=715, top=0)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis2.json')
    ctr1 = vis1.get_view_control()
    vis1.add_geometry(points_pcd_1)
    if key_tgt is not None:
        vis1.add_geometry(points_pcd_1_key)
    ctr1.convert_from_pinhole_camera_parameters(param)
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='geotrans_src_points', width=640, height=520, left=0, top=620)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis1.json')
    ctr2 = vis2.get_view_control()
    vis2.add_geometry(points_pcd_2)
    if key_src is not None:
        vis2.add_geometry(points_pcd_2_key)
    ctr2.convert_from_pinhole_camera_parameters(param)
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='geotrans_ref_points', width=640, height=520, left=715, top=620)
    param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis2.json')
    ctr3 = vis3.get_view_control()
    vis3.add_geometry(points_pcd_3)
    if key_tgt is not None:
        vis3.add_geometry(points_pcd_3_key)
    ctr3.convert_from_pinhole_camera_parameters(param)
    if cp_our is not None and cp_geotran is not None:
        # vis4 = o3d.visualization.Visualizer()
        # vis4.create_window(window_name='our_cp', width=640, height=520, left=1360, top=0)
        # vis4.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_vis4.json')
        # param = o3d.io.read_pinhole_camera_parameters('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis4.json')
        # ctr4 = vis4.get_view_control()
        # vis4.add_geometry(points_pcd_0_our)
        # vis4.add_geometry(points_pcd_1_our)
        # vis4.add_geometry(lineset_pcd_our)
        # ctr4.convert_from_pinhole_camera_parameters(param)
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name='geotrans_cp', width=640, height=520, left=1360, top=620)
        vis5.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/render_json/RenderOption_vis4.json')
        ctr5 = vis5.get_view_control()
        vis5.add_geometry(points_pcd_2_geo)
        vis5.add_geometry(points_pcd_3_geo)
        # vis5.add_geometry(lineset_pcd_geo)
        # vis5.add_geometry(lineset_pcd_geo_c)
        ctr5.convert_from_pinhole_camera_parameters(param)

    while True:
        vis0.update_geometry(points_pcd_0)
        if not vis0.poll_events():
            break
        vis0.update_renderer()
        vis1.update_geometry(points_pcd_1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
        vis2.update_geometry(points_pcd_2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
        # param = vis2.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('vis_tmp1.json', param)
        vis3.update_geometry(points_pcd_3)
        if not vis3.poll_events():
            break
        vis3.update_renderer()
        # param = vis3.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('vis_tmp2.json', param)

        if cp_our is not None and cp_geotran is not None:
            # vis4.update_geometry(points_pcd_0_our)
            # vis4.update_geometry(points_pcd_1_our)
            # vis4.update_geometry(lineset_pcd_our)
            # if not vis4.poll_events():
            #     break
            # vis4.update_renderer()
            # param = vis4.get_view_control().convert_to_pinhole_camera_parameters()
            # o3d.io.write_pinhole_camera_parameters('view1.json', param)

            vis5.update_geometry(points_pcd_2_geo)
            vis5.update_geometry(points_pcd_3_geo)
            # vis5.update_geometry(lineset_pcd_geo)
            # vis5.update_geometry(lineset_pcd_geo_c)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

    vis0.destroy_window()
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()


def draw_registration_est(points0, points1, points2, points3, our_tfms, geo_tfms, gt_tfms, ids, Note=None):
    ########################################
    # 1. input point cloud
    def create_correspondence(src_raw, tgt_raw, gt_tsfm):
        translate = [0, 0, 0]

        src_pcd = to_o3d_pcd(src_raw)
        tgt_pcd = to_o3d_pcd(tgt_raw)
        src_pcd.paint_uniform_color(get_yellow())
        tgt_pcd.paint_uniform_color(get_blue())
        src_pcd.transform(gt_tsfm)
        tgt_pcd.translate(translate)
        return src_pcd, tgt_pcd

    points_pcd_0_our, points_pcd_1_our = create_correspondence(points0, points1, our_tfms)
    points_pcd_2_geo, points_pcd_3_geo = create_correspondence(points2, points3, geo_tfms)
    points_pcd_4_gt, points_pcd_5_gt = create_correspondence(points2, points3, gt_tfms)

    ########################################
    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='our_cp', width=640, height=520, left=1360, top=0)
    vis4.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_vis4.json')
    param = o3d.io.read_pinhole_camera_parameters(f'/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/id_{ids}_view.json' if Note is None else f'/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/{Note}_id_{ids}_view.json')
    ctr4 = vis4.get_view_control()
    vis4.add_geometry(points_pcd_0_our)
    vis4.add_geometry(points_pcd_1_our)
    ctr4.convert_from_pinhole_camera_parameters(param)
    vis5 = o3d.visualization.Visualizer()
    vis5.create_window(window_name='geotrans_cp', width=640, height=520, left=1360, top=620)
    vis5.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_vis4.json')
    ctr5 = vis5.get_view_control()
    vis5.add_geometry(points_pcd_2_geo)
    vis5.add_geometry(points_pcd_3_geo)
    ctr5.convert_from_pinhole_camera_parameters(param)

    vis6 = o3d.visualization.Visualizer()
    vis6.create_window(window_name='gt_cp', width=640, height=520, left=715, top=0)
    vis6.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_vis4.json')
    ctr6 = vis6.get_view_control()
    vis6.add_geometry(points_pcd_4_gt)
    vis6.add_geometry(points_pcd_5_gt)
    ctr6.convert_from_pinhole_camera_parameters(param)

    while True:
        vis4.update_geometry(points_pcd_0_our)
        vis4.update_geometry(points_pcd_1_our)
        if not vis4.poll_events():
            break
        vis4.update_renderer()
        # param = vis4.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters(f'render_json/id_{ids}_view.json' if Note is None else f'render_json/{Note}_id_{ids}_view.json', param)

        vis5.update_geometry(points_pcd_2_geo)
        vis5.update_geometry(points_pcd_3_geo)
        if not vis5.poll_events():
            break
        vis5.update_renderer()

        vis6.update_geometry(points_pcd_4_gt)
        vis6.update_geometry(points_pcd_5_gt)
        if not vis6.poll_events():
            break
        vis6.update_renderer()

    vis4.destroy_window()
    vis5.destroy_window()
    vis6.destroy_window()


def draw_registration_gt(points0, points1, gt_tfms, ids, Note=None):
    ########################################
    # 1. input point cloud
    def create_correspondence(src_raw, tgt_raw, gt_tsfm):
        translate = [0, 0, 0]

        src_pcd = to_o3d_pcd(src_raw)
        tgt_pcd = to_o3d_pcd(tgt_raw)
        src_pcd.paint_uniform_color(get_yellow())
        tgt_pcd.paint_uniform_color(get_blue())
        src_pcd.transform(gt_tsfm)
        tgt_pcd.translate(translate)
        return src_pcd, tgt_pcd

    points_pcd_4_gt, points_pcd_5_gt = create_correspondence(points0, points1, gt_tfms)

    ########################################
    param = o3d.io.read_pinhole_camera_parameters(f'/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis1.json')
    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='ref', width=640, height=520, left=715, top=0)
    vis4.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_vis4.json')
    ctr4 = vis4.get_view_control()
    vis4.add_geometry(points_pcd_4_gt)
    ctr4.convert_from_pinhole_camera_parameters(param)

    param = o3d.io.read_pinhole_camera_parameters(f'/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/vis2.json')
    vis5 = o3d.visualization.Visualizer()
    vis5.create_window(window_name='src', width=640, height=520, left=715, top=620)
    vis5.get_render_option().load_from_json('/home/science/code/python3/GeoTransformer/experiments/3DMatch/render_json/RenderOption_vis4.json')
    ctr5 = vis5.get_view_control()
    vis5.add_geometry(points_pcd_5_gt)
    ctr5.convert_from_pinhole_camera_parameters(param)

    while True:
        vis4.update_geometry(points_pcd_4_gt)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

        vis5.update_geometry(points_pcd_5_gt)
        if not vis5.poll_events():
            break
        vis5.update_renderer()

    vis4.destroy_window()
    vis5.destroy_window()


def draw_single_attention(points, nodes, nodes_keys, nodes_val):
    ########################################
    # 1. input point cloud
    points = to_array(points)
    nodes = to_array(nodes)
    nodes_keys = to_array(nodes_keys)
    nodes_val = to_array(nodes_val)

    points_pcd = to_o3d_pcd(points)
    points_pcd.paint_uniform_color(get_blue())
    points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    nodes_pcd = to_o3d_pcd(nodes)
    nodes_pcd.paint_uniform_color(get_yellow())
    nodes_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    nodes_keypoints = nodes[nodes_keys, :]
    nodes_keypoints_pcd = to_o3d_pcd(nodes_keypoints)
    nodes_keypoints_pcd.paint_uniform_color(get_red())
    nodes_keypoints_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    # set point size
    nodes_keypoints_pcd.scale(0.01, center=nodes_keypoints_pcd.get_center())

    # nodes_val = np.repeat(nodes_val.mean(0).transpose(1, 0), 3, axis=1)
    # nodes_key_color = lighter(get_blue(), 1 - nodes_val)
    # nodes_pcd.colors = o3d.utility.Vector3dVector(nodes_key_color)

    ########################################
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(points_pcd)
    vis1.add_geometry(nodes_pcd)
    vis1.add_geometry(nodes_keypoints_pcd)

    while True:
        vis1.update_geometry(points_pcd)
        vis1.update_geometry(nodes_pcd)
        vis1.update_geometry(nodes_keypoints_pcd)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_single_attention_1(points, nodes):
    ########################################
    # 1. input point cloud
    points = to_array(points)
    nodes = to_array(nodes)

    points_pcd = to_o3d_pcd(points)
    points_pcd.paint_uniform_color(lighter(get_blue(),0.8))
    points_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    nodes_pcd = to_o3d_pcd(nodes)
    nodes_pcd.paint_uniform_color(get_red())
    nodes_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(points_pcd)
    vis1.add_geometry(nodes_pcd)


    while True:
        vis1.update_geometry(points_pcd)
        vis1.update_geometry(nodes_pcd)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_show_attention(src_raw, tgt_raw, est_tsfm=np.eye(4), attention_score_src=None, attention_score_tgt=None):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_overlap = attention_score_src[: ,None].repeat(1 ,3).numpy()
    tgt_overlap = attention_score_tgt[: ,None].repeat(1 ,3).numpy()
    src_overlap_color = lighter(get_yellow(), 1 - src_overlap)
    tgt_overlap_color = lighter(get_blue(), 1 - tgt_overlap)
    src_pcd_before.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_before.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    ########################################
    # 2. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_est_after)
    vis1.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, est_tsfm, gt_tsfm, title='', cp=None, inlier_distance_threshold=0.1):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. overlap colors
    rot, trans = to_tensor(est_tsfm[:3 , :3]), to_tensor(est_tsfm[:3 , 3][: , None])
    src_overlap = src_overlap[: ,None].repeat(1 ,3).numpy()
    tgt_overlap = tgt_overlap[: ,None].repeat(1 ,3).numpy()
    src_overlap_color = lighter(get_yellow(), 1 - src_overlap)
    tgt_overlap_color = lighter(get_blue(), 1 - tgt_overlap)
    src_pcd_est_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_est_overlap.transform(est_tsfm)
    tgt_pcd_est_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_est_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_est_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    src_pcd_gt_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_gt_overlap.transform(gt_tsfm)
    tgt_pcd_gt_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_gt_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_gt_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    ########################################
    # 3. compute cp line
    if cp is not None:
        translate = [-1.3, -1.5, 0]
        gt_inliers = get_inlier(src_raw[cp[:, 0], :], tgt_raw[cp[:, 1], :], gt_tsfm, inlier_distance_threshold)
        cp_temp = cp + np.array([0, src_raw.shape[0]])
        colors = [[0, 1, 0] if i in gt_inliers else [1, 0, 0] for i in range(cp.shape[0])]
        # cp_temp = cp[gt_inliers, :] + np.array([0, src_raw.shape[0]])
        line_set = o3d.geometry.LineSet()
        src_pcd_cp = to_o3d_pcd(src_raw)
        tgt_pcd_cp = to_o3d_pcd(tgt_raw)
        src_pcd_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_cp.points,tgt_pcd_cp.points]))
        line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        line_set.colors = o3d.utility.Vector3dVector(colors)

    ########################################
    # 4. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)

    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name=f'Inferred {title} region for est', width=640, height=520, left=640, top=0)
    vis2.add_geometry(src_pcd_est_overlap)
    vis2.add_geometry(tgt_pcd_est_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name ='Our registration', width=640, height=520, left=1280, top=0)
    vis3.add_geometry(src_pcd_est_after)
    vis3.add_geometry(tgt_pcd_before)

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name ='Gt registration', width=640, height=520, left=1280, top=570)
    vis4.add_geometry(src_pcd_gt_after)
    vis4.add_geometry(tgt_pcd_before)

    vis5 = o3d.visualization.Visualizer()
    vis5.create_window(window_name=f'Inferred {title} region for gt', width=640, height=520, left=640, top=570)
    vis5.add_geometry(src_pcd_gt_overlap)
    vis5.add_geometry(tgt_pcd_gt_overlap)

    if cp is not None:
        vis6 = o3d.visualization.Visualizer()
        vis6.create_window(window_name=f'cp {title}', width=640, height=520, left=0, top=570)
        vis6.add_geometry(src_pcd_cp)
        vis6.add_geometry(tgt_pcd_cp)
        vis6.add_geometry(line_set)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_est_overlap)
        vis2.update_geometry(tgt_pcd_est_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_est_after)
        vis3.update_geometry(tgt_pcd_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        vis4.update_geometry(src_pcd_gt_after)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

        vis5.update_geometry(src_pcd_gt_overlap)
        vis5.update_geometry(tgt_pcd_gt_overlap)
        if not vis5.poll_events():
            break
        vis5.update_renderer()

        if cp is not None:
            vis6.update_geometry(src_pcd_cp)
            vis6.update_geometry(tgt_pcd_cp)
            vis6.update_geometry(line_set)
            if not vis6.poll_events():
                break
            vis6.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()
    vis4.destroy_window()
    vis5.destroy_window()
    if cp is not None:
        vis6.destroy_window()


def draw_registration_pcpair(src_raw, tgt_raw, est_tsfm=np.eye(4), title='', cp=None):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_est_after)
    vis1.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_registration_2pcpair(src_raw, tgt_raw, gt_tsfm, est_tsfm, init_tsfm=None, title=''):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    gt_tsfm = to_array(gt_tsfm)
    est_tsfm = to_array(est_tsfm)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)
    if init_tsfm is not None:
        init_tsfm = to_array(init_tsfm)
        src_pcd_est_after_init = copy.deepcopy(src_pcd_before)
        src_pcd_est_after_init.transform(init_tsfm)
    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Before registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='est final registration', width=640, height=520, left=640, top=0)
    vis2.add_geometry(src_pcd_est_after)
    vis2.add_geometry(tgt_pcd_before)

    if init_tsfm is not None:
        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name='est init registration', width=640, height=520, left=640, top=570)
        vis3.add_geometry(src_pcd_est_after_init)
        vis3.add_geometry(tgt_pcd_before)

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='gt registration', width=640, height=520, left=1280, top=0)
    vis4.add_geometry(src_pcd_gt_after)
    vis4.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_est_after)
        vis2.update_geometry(tgt_pcd_before)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        if init_tsfm is not None:
            vis3.update_geometry(src_pcd_est_after_init)
            vis3.update_geometry(tgt_pcd_before)
            if not vis3.poll_events():
                break
            vis3.update_renderer()

        vis4.update_geometry(src_pcd_gt_after)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    if init_tsfm is not None:
        vis3.destroy_window()
    vis4.destroy_window()


def draw_registration_2pcpairmuch(src_raw, tgt_raw, gt_tsfm=None, est1_tsfm=None, est2_tsfm=None, est3_tsfm=None, est4_tsfm=None,
                                  title1='est 1', title2='est 2', title3='est 3', title4='est 4'):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    gt_tsfm = to_array(gt_tsfm)
    if est1_tsfm is not None:
        est1_tsfm = to_array(est1_tsfm)
    if est2_tsfm is not None:
        est2_tsfm = to_array(est2_tsfm)
    if est3_tsfm is not None:
        est3_tsfm = to_array(est3_tsfm)
    if est4_tsfm is not None:
        est4_tsfm = to_array(est4_tsfm)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)
    if est1_tsfm is not None:
        src_pcd_est1_after = copy.deepcopy(src_pcd_before)
        src_pcd_est1_after.transform(est1_tsfm)
    if est2_tsfm is not None:
        src_pcd_est2_after = copy.deepcopy(src_pcd_before)
        src_pcd_est2_after.transform(est2_tsfm)
    if est3_tsfm is not None:
        src_pcd_est3_after = copy.deepcopy(src_pcd_before)
        src_pcd_est3_after.transform(est3_tsfm)
    if est4_tsfm is not None:
        src_pcd_est4_after = copy.deepcopy(src_pcd_before)
        src_pcd_est4_after.transform(est4_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Before registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)


    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='gt registration', width=640, height=520, left=0, top=570)
    vis2.add_geometry(src_pcd_gt_after)
    vis2.add_geometry(tgt_pcd_before)

    if est1_tsfm is not None:
        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name=title1, width=640, height=520, left=640, top=0)
        vis3.add_geometry(src_pcd_est1_after)
        vis3.add_geometry(tgt_pcd_before)

    if est2_tsfm is not None:
        vis4 = o3d.visualization.Visualizer()
        vis4.create_window(window_name=title2, width=640, height=520, left=1280, top=0)
        vis4.add_geometry(src_pcd_est2_after)
        vis4.add_geometry(tgt_pcd_before)

    if est3_tsfm is not None:
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name=title3, width=640, height=520, left=640, top=570)
        vis5.add_geometry(src_pcd_est3_after)
        vis5.add_geometry(tgt_pcd_before)

    if est4_tsfm is not None:
        vis6 = o3d.visualization.Visualizer()
        vis6.create_window(window_name=title4, width=640, height=520, left=1280, top=570)
        vis6.add_geometry(src_pcd_est4_after)
        vis6.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_gt_after)
        vis2.update_geometry(tgt_pcd_before)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        if est1_tsfm is not None:
            vis3.update_geometry(src_pcd_est1_after)
            vis3.update_geometry(tgt_pcd_before)
            if not vis3.poll_events():
                break
            vis3.update_renderer()

        if est2_tsfm is not None:
            vis4.update_geometry(src_pcd_est2_after)
            vis4.update_geometry(tgt_pcd_before)
            if not vis4.poll_events():
                break
            vis4.update_renderer()

        if est3_tsfm is not None:
            vis5.update_geometry(src_pcd_est3_after)
            vis5.update_geometry(tgt_pcd_before)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

        if est4_tsfm is not None:
            vis6.update_geometry(src_pcd_est4_after)
            vis6.update_geometry(tgt_pcd_before)
            if not vis6.poll_events():
                break
            vis6.update_renderer()


    vis1.destroy_window()
    vis2.destroy_window()
    if est1_tsfm is not None:
        vis3.destroy_window()
    if est2_tsfm is not None:
        vis4.destroy_window()
    if est3_tsfm is not None:
        vis5.destroy_window()
    if est4_tsfm is not None:
        vis6.destroy_window()


def draw_registration_2pcpair_ds(src_raw, tgt_raw, src_sparse, tgt_sparse, gt_tsfm, est_tsfm=np.eye(4), title='', cp=None):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_sparse_before = to_o3d_pcd(src_sparse)
    tgt_pcd_sparse_before = to_o3d_pcd(tgt_sparse)
    src_pcd_sparse_before.paint_uniform_color(get_yellow())
    tgt_pcd_sparse_before.paint_uniform_color(get_blue())

    ########################################
    # 3. compute cp line
    if cp is not None:
        translate = [-1.3, -1.5, 0]
        gt_inliers = get_inlier(src_sparse[cp[:, 0], :], tgt_sparse[cp[:, 1], :], gt_tsfm)
        est_inliers = get_inlier(src_sparse[cp[:, 0], :], tgt_sparse[cp[:, 1], :], est_tsfm)
        cp_temp = cp + np.array([0, src_sparse.shape[0]])
        gt_colors = [[0, 1, 0] if i in gt_inliers else [1, 0, 0] for i in range(cp.shape[0])]
        est_colors = [[0, 1, 0] if i in est_inliers else [1, 0, 0] for i in range(cp.shape[0])]
        # cp_temp = cp[gt_inliers, :] + np.array([0, src_raw.shape[0]])
        gt_line_set = o3d.geometry.LineSet()
        est_line_set = o3d.geometry.LineSet()
        src_pcd_gt_cp = to_o3d_pcd(src_sparse)
        src_pcd_est_cp = to_o3d_pcd(src_sparse)
        tgt_pcd_cp = to_o3d_pcd(tgt_sparse)
        src_pcd_gt_cp.paint_uniform_color(get_yellow())
        src_pcd_est_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_gt_cp.transform(gt_tsfm)
        src_pcd_est_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        gt_line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_gt_cp.points,tgt_pcd_cp.points]))
        gt_line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        gt_line_set.colors = o3d.utility.Vector3dVector(gt_colors)

        est_line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_est_cp.points,tgt_pcd_cp.points]))
        est_line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        est_line_set.colors = o3d.utility.Vector3dVector(est_colors)

    ########################################
    # 2. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)
    src_pcd_sparse_est_after = copy.deepcopy(src_pcd_sparse_before)
    src_pcd_sparse_est_after.transform(est_tsfm)
    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)
    src_pcd_sparse_gt_after = copy.deepcopy(src_pcd_sparse_before)
    src_pcd_sparse_gt_after.transform(gt_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Before registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='After registration', width=640, height=520, left=640, top=0)
    vis2.add_geometry(src_pcd_est_after)
    vis2.add_geometry(tgt_pcd_before)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='After registration', width=640, height=520, left=1280, top=0)
    vis3.add_geometry(src_pcd_sparse_est_after)
    vis3.add_geometry(tgt_pcd_sparse_before)

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='gt registration', width=640, height=520, left=640, top=570)
    vis4.add_geometry(src_pcd_gt_after)
    vis4.add_geometry(tgt_pcd_before)

    if cp is not None:
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name=f'gt cp', width=640, height=520, left=1280, top=570)
        vis5.add_geometry(src_pcd_gt_cp)
        vis5.add_geometry(tgt_pcd_cp)
        vis5.add_geometry(gt_line_set)

        vis6 = o3d.visualization.Visualizer()
        vis6.create_window(window_name=f'est cp', width=640, height=520, left=0, top=570)
        vis6.add_geometry(src_pcd_est_cp)
        vis6.add_geometry(tgt_pcd_cp)
        vis6.add_geometry(est_line_set)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_est_after)
        vis2.update_geometry(tgt_pcd_before)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_sparse_est_after)
        vis3.update_geometry(tgt_pcd_sparse_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        vis4.update_geometry(src_pcd_gt_after)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

        if cp is not None:
            vis5.update_geometry(src_pcd_gt_cp)
            vis5.update_geometry(tgt_pcd_cp)
            vis5.update_geometry(gt_line_set)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

            vis6.update_geometry(src_pcd_est_cp)
            vis6.update_geometry(tgt_pcd_cp)
            vis6.update_geometry(est_line_set)
            if not vis6.poll_events():
                break
            vis6.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()
    vis4.destroy_window()
    if cp is not None:
        vis5.destroy_window()
        vis6.destroy_window()


def draw_registration_s2d_pcpair(src_raw, tgt_raw, src_down, tgt_down, est_tsfm=np.eye(4), title='', cp=None):
    # show those sparse and dense point clouds
    ########################################
    # 1. input point cloud
    src_pcd_raw_before = to_o3d_pcd(src_raw)
    tgt_pcd_raw_before = to_o3d_pcd(tgt_raw)
    src_pcd_raw_before.paint_uniform_color(lighter(get_yellow(), 0.5))
    tgt_pcd_raw_before.paint_uniform_color(lighter(get_blue(), 0.9))
    src_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_down_before = to_o3d_pcd(src_down)
    tgt_pcd_down_before = to_o3d_pcd(tgt_down)
    src_pcd_down_before.paint_uniform_color(get_yellow_down())
    tgt_pcd_down_before.paint_uniform_color(get_blue_down())
    src_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_raw_est_after = copy.deepcopy(src_pcd_raw_before)
    src_pcd_raw_est_after.transform(est_tsfm)
    src_pcd_down_est_after = copy.deepcopy(src_pcd_down_before)
    src_pcd_down_est_after.transform(est_tsfm)


    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_raw_est_after)
    vis1.add_geometry(tgt_pcd_raw_before)
    vis1.add_geometry(src_pcd_down_est_after)
    vis1.add_geometry(tgt_pcd_down_before)

    while True:
        vis1.update_geometry(src_pcd_raw_est_after)
        vis1.update_geometry(tgt_pcd_raw_before)
        vis1.update_geometry(src_pcd_down_est_after)
        vis1.update_geometry(tgt_pcd_down_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_registration_sphere_pcpair(src_raw, tgt_raw, src_down, tgt_down, src_down_sp, tgt_down_sp, est_tsfm=np.eye(4), title='', cp=None):
    # show those sparse and dense point clouds
    ########################################
    # 1. input point cloud
    src_pcd_raw_before = to_o3d_pcd(src_raw)
    tgt_pcd_raw_before = to_o3d_pcd(tgt_raw)
    src_pcd_raw_before.paint_uniform_color(get_yellow())
    tgt_pcd_raw_before.paint_uniform_color(get_blue())
    src_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_down_before = to_o3d_pcd(src_down)
    tgt_pcd_down_before = to_o3d_pcd(tgt_down)
    src_pcd_down_before.paint_uniform_color(get_yellow_down())
    tgt_pcd_down_before.paint_uniform_color(get_blue_down())
    src_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_down_sp_before = to_o3d_pcd(src_down_sp)
    tgt_pcd_down_sp_before = to_o3d_pcd(tgt_down_sp)
    src_pcd_down_sp_before.paint_uniform_color(get_yellow_down())
    tgt_pcd_down_sp_before.paint_uniform_color(get_blue_down())
    src_pcd_down_sp_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_down_sp_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_raw_est_after = copy.deepcopy(src_pcd_raw_before)
    src_pcd_raw_est_after.transform(est_tsfm)
    src_pcd_down_est_after = copy.deepcopy(src_pcd_down_before)
    src_pcd_down_est_after.transform(est_tsfm)
    src_pcd_down_sp_est_after = copy.deepcopy(src_pcd_down_sp_before)
    src_pcd_down_sp_est_after.transform(est_tsfm)


    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_raw_est_after)
    vis1.add_geometry(tgt_pcd_raw_before)
    vis1.add_geometry(src_pcd_down_est_after)
    vis1.add_geometry(tgt_pcd_down_before)
    vis1.add_geometry(src_pcd_down_sp_est_after)
    vis1.add_geometry(tgt_pcd_down_sp_before)

    while True:
        vis1.update_geometry(src_pcd_raw_est_after)
        vis1.update_geometry(tgt_pcd_raw_before)
        vis1.update_geometry(src_pcd_down_est_after)
        vis1.update_geometry(tgt_pcd_down_before)
        vis1.update_geometry(src_pcd_down_sp_est_after)
        vis1.update_geometry(tgt_pcd_down_sp_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()

def draw_registration_2pcpair_cp(src_sparse, tgt_sparse, gt_tsfm, cp=None, title='', inlier_th=0.05, translate_vec=[0,0,-0.5]):
    ########################################
    # 1. input point cloud
    src_pcd_sparse_before = to_o3d_pcd(src_sparse)
    tgt_pcd_sparse_before = to_o3d_pcd(tgt_sparse)
    src_pcd_sparse_before.paint_uniform_color(get_yellow())
    tgt_pcd_sparse_before.paint_uniform_color(get_blue())

    ########################################
    # 3. compute cp line
    if cp is not None:
        translate = translate_vec
        gt_inliers = get_inlier(src_sparse[cp[:, 0], :], tgt_sparse[cp[:, 1], :], gt_tsfm, inlier_distance_threshold=inlier_th)
        cp_temp = cp + np.array([0, src_sparse.shape[0]])
        gt_colors = [[0, 1, 0] if i in gt_inliers else [1, 0, 0] for i in range(cp.shape[0])]
        # cp_temp = cp[gt_inliers, :] + np.array([0, src_raw.shape[0]])
        gt_line_set = o3d.geometry.LineSet()
        src_pcd_gt_cp = to_o3d_pcd(src_sparse)
        tgt_pcd_cp = to_o3d_pcd(tgt_sparse)
        src_pcd_gt_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_gt_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        gt_line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_gt_cp.points,tgt_pcd_cp.points]))
        gt_line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        gt_line_set.colors = o3d.utility.Vector3dVector(gt_colors)

    ########################################
    # 2. draw registrations
    src_pcd_sparse_gt_after = copy.deepcopy(src_pcd_sparse_before)
    src_pcd_sparse_gt_after.transform(gt_tsfm)


    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='Reg '+ title, width=640, height=520, left=1280, top=0)
    vis3.add_geometry(src_pcd_sparse_gt_after)
    vis3.add_geometry(tgt_pcd_sparse_before)

    if cp is not None:
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name='CP '+ title, width=640, height=520, left=1280, top=570)
        vis5.add_geometry(src_pcd_gt_cp)
        vis5.add_geometry(tgt_pcd_cp)
        vis5.add_geometry(gt_line_set)

    while True:

        vis3.update_geometry(src_pcd_sparse_gt_after)
        vis3.update_geometry(tgt_pcd_sparse_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        if cp is not None:
            vis5.update_geometry(src_pcd_gt_cp)
            vis5.update_geometry(tgt_pcd_cp)
            vis5.update_geometry(gt_line_set)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

    vis3.destroy_window()
    if cp is not None:
        vis5.destroy_window()
