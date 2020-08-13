# 1. Voxel Content
* Denote V as a non-empty voxel, containing $t \leq T$ points

$$
V=\{p_i= \left[ x_i, y_i, z_i, r_i\right]^T \}_{i=1,\dots,t}
$$

* calculate centroid point $\hat{p} = (v_x, v_y, v_z)$
* augment each point $p_i$ with relative offset w.r.t centroid point, similar to 2D anchor based approach

$$
V_{in}=\{p_i= \left[ x_i, y_i, z_i, r_i ,x_i - v_x, y_i - v_y, z_i - v_z  \right]^T \}_{i=1,\dots,t}
$$

# 2. Encoding like SSD
* anchors $d:=[x_a,y_a,z_a,l_a,w_a,h_a,r_a]$
* boxes $b:=[x_g,y_g,z_g,l_g,w_g,h_g,r_g]$
  * $g$ stands for ground truth
* learned value from nn during trainig is $l$, the difference from $d$ to $b$, denote as $l:=[x_l,y_l,z_l,l_l,w_l,h_l,r_l]$
  * $l$ stands for learned
* encode for learning process in `box_torch.ops.py`

```python
diagonal = torch.sqrt(la ** 2 + wa ** 2)
xt = (xg - xa) / diagonal
yt = (yg - ya) / diagonal
zt = (zg - za) / ha
cts = [g - a for g, a in zip(cgs, cas)]
if smooth_dim:
    lt = lg / la - 1
    wt = wg / wa - 1
    ht = hg / ha - 1
else:
    lt = torch.log(lg / la)
    wt = torch.log(wg / wa)
    ht = torch.log(hg / ha)
if encode_angle_to_vector:
    rgx = torch.cos(rg)
    rgy = torch.sin(rg)
    rax = torch.cos(ra)
    ray = torch.sin(ra)
    rtx = rgx - rax
    rty = rgy - ray
    return torch.cat([xt, yt, zt, wt, lt, ht, rtx, rty, *cts], dim=-1)
else:
    rt = rg - ra
    return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)
```

* decode for predict process