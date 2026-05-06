# esegformer3d.R
#
# R torch port of TreeAIBox's vox3DESegFormer.py
# (https://github.com/NRCan/TreeAIBox/blob/main/modules/filter/vox3DESegFormer.py)
#
# Original: Zhouxin Xi (NRCan), CC BY-NC 4.0 for trained weights.
#
# Differences from vox3DSegFormer.py / segformer3d.R:
#   - LayerNorm replaced by DyT (Dynamic Tanh with learnable gamma/beta)
#   - Attention replaced by EfficientAttention (separate q/k/v projections,
#     uses F.scaled_dot_product_attention equivalent)
#   - Decoder uses EfficientSkipConnection instead of concatenation + conv
#   - MLP uses hidden_dim // 2 (memory-efficient Mlp)
#   - Two-stage linear_pred: Conv3d(dec,dec//2,3,pad=1) -> GELU -> Conv3d(dec//2,nc,1)
#
# Layer-name contract with upstream state_dict keys:
#   patch_embed{1..4}.proj.{weight,bias}
#   patch_embed{1..4}.norm.{alpha,weight,bias}    <- DyT params
#   block{1..4}.{i}.norm1.{alpha,weight,bias}
#   block{1..4}.{i}.attn.q.{weight,bias}
#   block{1..4}.{i}.attn.k.{weight,bias}
#   block{1..4}.{i}.attn.v.{weight,bias}
#   block{1..4}.{i}.attn.proj.{weight,bias}
#   block{1..4}.{i}.attn.proj_drop  (no params)
#   block{1..4}.{i}.attn.norm.{alpha,weight,bias}
#   block{1..4}.{i}.attn.sr.{weight,bias}          (sr_ratio > 1)
#   block{1..4}.{i}.attn.sr_norm.{alpha,weight,bias} (sr_ratio > 1)
#   block{1..4}.{i}.norm2.{alpha,weight,bias}
#   block{1..4}.{i}.mlp.fc1.{weight,bias}
#   block{1..4}.{i}.mlp.dwconv.dwconv.{weight,bias}
#   block{1..4}.{i}.mlp.fc2.{weight,bias}
#   norm{1..4}.{alpha,weight,bias}
#   linear_c{1..4}.proj.{weight,bias}
#   skip_fusions.{0..2}.fusion.0.{weight,bias}
#   linear_fuse.{weight,bias}
#   linear_pred.0.{weight,bias}
#   linear_pred.2.{weight,bias}

# --------------------------------------------------------------------------
# DyT — Dynamic Tanh (replaces LayerNorm throughout)
# --------------------------------------------------------------------------
DyT <- nn_module(
  classname = "DyT",
  initialize = function(num_features, alpha_init_value = 0.5) {
    self$alpha  <- nn_parameter(torch_ones(1) * alpha_init_value)
    self$weight <- nn_parameter(torch_ones(num_features))
    self$bias   <- nn_parameter(torch_zeros(num_features))
  },
  forward = function(x) {
    x <- torch_tanh(self$alpha * x)
    x * self$weight + self$bias
  }
)

# --------------------------------------------------------------------------
# EfficientAttention — separate q/k/v, uses sdp attention
# --------------------------------------------------------------------------
EfficientAttention <- nn_module(
  classname = "EfficientAttention",
  initialize = function(dim, num_heads = 8, qkv_bias = FALSE,
                        qk_scale = NULL, attn_drop = 0, proj_drop = 0,
                        sr_ratio = 1) {
    if (dim %% num_heads != 0)
      stop(sprintf("dim %d must be divisible by num_heads %d", dim, num_heads))
    self$dim       <- dim
    self$num_heads <- num_heads
    head_dim       <- dim %/% num_heads
    self$scale     <- qk_scale %||% (head_dim ^ -0.5)

    self$q         <- nn_linear(dim, dim, bias = qkv_bias)
    self$k         <- nn_linear(dim, dim, bias = qkv_bias)
    self$v         <- nn_linear(dim, dim, bias = qkv_bias)
    self$attn_drop <- nn_dropout(attn_drop)
    self$proj      <- nn_linear(dim, dim)
    self$proj_drop <- nn_dropout(proj_drop)
    self$norm      <- DyT(dim)

    self$sr_ratio <- sr_ratio
    if (sr_ratio > 1) {
      self$sr      <- nn_conv3d(dim, dim, kernel_size = sr_ratio,
                                stride = sr_ratio)
      self$sr_norm <- DyT(dim)
    }
  },
  forward = function(x, D, H, W) {
    sz <- x$size()
    B <- sz[1]; N <- sz[2]; C <- sz[3]
    H_ <- self$num_heads
    Cph <- C %/% H_

    q <- self$q(x)$reshape(c(B, N, H_, Cph))$permute(c(1, 3, 2, 4))

    if (self$sr_ratio > 1) {
      x_ <- x$permute(c(1, 3, 2))$reshape(c(B, C, D, H, W))
      x_ <- self$sr(x_)$reshape(c(B, C, -1L))$permute(c(1, 3, 2))
      x_ <- self$sr_norm(x_)
      k  <- self$k(x_)$reshape(c(B, -1L, H_, Cph))$permute(c(1, 3, 2, 4))
      v  <- self$v(x_)$reshape(c(B, -1L, H_, Cph))$permute(c(1, 3, 2, 4))
    } else {
      k <- self$k(x)$reshape(c(B, -1L, H_, Cph))$permute(c(1, 3, 2, 4))
      v <- self$v(x)$reshape(c(B, -1L, H_, Cph))$permute(c(1, 3, 2, 4))
    }

    # Scale q then compute attention manually (R torch may lack F.sdpa)
    q <- q * self$scale
    attn <- q$matmul(k$transpose(-2L, -1L))
    attn <- nnf_softmax(attn, dim = -1L)
    if (self$attn_drop$p > 0 && self$training)
      attn <- self$attn_drop(attn)

    out <- attn$matmul(v)$transpose(2L, 3L)$reshape(c(B, N, C))
    out <- self$proj(out)
    out <- self$proj_drop(out)
    self$norm(out)
  }
)

# --------------------------------------------------------------------------
# Mlp (memory-efficient: hidden_dim // 2, same DWConv backbone)
# --------------------------------------------------------------------------
EfficientMlp <- nn_module(
  classname = "Mlp",             # must match upstream "Mlp" class name
  initialize = function(in_features, hidden_features = NULL,
                        out_features = NULL, drop = 0) {
    out_features    <- out_features    %||% in_features
    hidden_features <- hidden_features %||% in_features
    # upstream halves the hidden dim for memory efficiency
    reduced <- max(as.integer(hidden_features %/% 2L), 32L)
    self$fc1    <- nn_linear(in_features, reduced)
    self$dwconv <- DWConv(reduced)
    self$act    <- nn_gelu()
    self$fc2    <- nn_linear(reduced, out_features)
    self$drop   <- nn_dropout(drop)
  },
  forward = function(x, D, H, W) {
    x <- self$fc1(x)
    x <- self$dwconv(x, D, H, W)
    x <- self$act(x)
    x <- self$drop(x)
    x <- self$fc2(x)
    self$drop(x)
  }
)

# --------------------------------------------------------------------------
# EBlock — Transformer block using DyT norms + EfficientAttention
# --------------------------------------------------------------------------
EBlock <- nn_module(
  classname = "Block",           # upstream calls this "Block" in both variants
  initialize = function(dim, num_heads, mlp_ratio = 4, qkv_bias = FALSE,
                        qk_scale = NULL, drop = 0, attn_drop = 0,
                        drop_path = 0, sr_ratio = 1) {
    self$norm1 <- DyT(dim)
    self$attn  <- EfficientAttention(
      dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale,
      attn_drop = attn_drop, proj_drop = drop, sr_ratio = sr_ratio)
    self$drop_path <- if (drop_path > 0) nn_drop_path(drop_path) else nn_identity()
    self$norm2 <- DyT(dim)
    self$mlp   <- EfficientMlp(
      in_features     = dim,
      hidden_features = as.integer(dim * mlp_ratio),
      drop            = drop)
  },
  forward = function(x, D, H, W) {
    x <- x + self$drop_path(self$attn(self$norm1(x), D, H, W))
    x + self$drop_path(self$mlp(self$norm2(x), D, H, W))
  }
)

# --------------------------------------------------------------------------
# OverlapPatchEmbed with DyT norm (replaces LayerNorm in upstream ESegFormer)
# --------------------------------------------------------------------------
EOverlapPatchEmbed <- nn_module(
  classname = "OverlapPatchEmbed",
  initialize = function(block3d_size = 224, patch_size = 7, stride = 4,
                        in_chans = 1, embed_dim = 768) {
    block3d_size <- to_3tuple(block3d_size)
    patch_size   <- to_3tuple(patch_size)
    self$proj <- nn_conv3d(in_chans, embed_dim,
                           kernel_size = patch_size,
                           stride      = stride,
                           padding     = c(patch_size[1] %/% 2,
                                           patch_size[2] %/% 2,
                                           patch_size[3] %/% 2))
    self$norm <- DyT(embed_dim)
  },
  forward = function(x) {
    x  <- self$proj(x)
    sz <- x$size()
    D <- sz[3]; H <- sz[4]; W <- sz[5]
    x <- x$flatten(start_dim = 3)$transpose(2L, 3L)
    list(x = self$norm(x), D = D, H = H, W = W)
  }
)

# --------------------------------------------------------------------------
# EfficientSkipConnection — concat + 1x1x1 conv + GELU
# --------------------------------------------------------------------------
EfficientSkipConnection <- nn_module(
  classname = "EfficientSkipConnection",
  initialize = function(decoder_dim) {
    self$fusion <- nn_sequential(
      nn_conv3d(decoder_dim * 2L, decoder_dim, kernel_size = 1L),
      nn_gelu()
    )
  },
  forward = function(x, skip) {
    self$fusion(torch_cat(list(x, skip), dim = 2L))
  }
)

# --------------------------------------------------------------------------
# Top-level ESegFormer3D model
# Supports three head types that mirror the three upstream Python variants:
#   "segmentation"    (default) – vox3DESegFormer.py  (woodcls, stemcls)
#   "detection_stem"            – vox3DSegFormerDetection.py with if_stem=TRUE
#                                 (treeloc TLS/boreal); 2D Conv2d prediction head
#   "regression"                – vox3DSegFormerRegression.py (treeoff); 3-stage
#                                 Conv3d head outputting out_chans channels
# --------------------------------------------------------------------------
ESegformer3D <- nn_module(
  classname = "Segformer",       # upstream class name is still "Segformer"
  initialize = function(block3d_size   = 1024,
                        patch_size     = 3,
                        in_chans       = 1,
                        num_classes    = 3,
                        embed_dims     = c(32, 64, 128, 256),
                        num_heads      = c(1, 2, 4, 8),
                        mlp_ratios     = c(2, 2, 2, 2),
                        qkv_bias       = TRUE,
                        qk_scale       = NULL,
                        drop_rate      = 0,
                        attn_drop_rate = 0,
                        drop_path_rate = 0,
                        depths         = c(2, 2, 8, 2),
                        sr_ratios      = c(8, 4, 2, 1),
                        decoder_dim    = 128,
                        head_type      = "segmentation",
                        out_chans      = 2L) {
    self$num_classes <- num_classes
    self$head_type   <- head_type
    self$depths      <- depths

    # Patch embeddings (all use DyT norm)
    self$patch_embed1 <- EOverlapPatchEmbed(
      block3d_size = block3d_size,
      patch_size   = patch_size, stride = 2L,
      in_chans     = in_chans,   embed_dim = embed_dims[1])
    self$patch_embed2 <- EOverlapPatchEmbed(
      block3d_size = block3d_size %/% 4L,
      patch_size   = patch_size, stride = 2L,
      in_chans     = embed_dims[1], embed_dim = embed_dims[2])
    self$patch_embed3 <- EOverlapPatchEmbed(
      block3d_size = block3d_size %/% 8L,
      patch_size   = patch_size, stride = 2L,
      in_chans     = embed_dims[2], embed_dim = embed_dims[3])
    self$patch_embed4 <- EOverlapPatchEmbed(
      block3d_size = block3d_size %/% 16L,
      patch_size   = patch_size, stride = 2L,
      in_chans     = embed_dims[3], embed_dim = embed_dims[4])

    # Stochastic depth schedule
    dpr <- as.numeric(torch_linspace(0, drop_path_rate, sum(depths)))
    cur <- 0L

    make_stage <- function(stage_idx) {
      blks <- lapply(seq_len(depths[stage_idx]), function(i) {
        EBlock(dim       = embed_dims[stage_idx],
               num_heads = num_heads[stage_idx],
               mlp_ratio = mlp_ratios[stage_idx],
               qkv_bias  = qkv_bias, qk_scale = qk_scale,
               drop      = drop_rate,
               attn_drop = attn_drop_rate,
               drop_path = dpr[cur + i],
               sr_ratio  = sr_ratios[stage_idx])
      })
      nn_module_list(blks)
    }

    self$block1 <- make_stage(1L); cur <- cur + depths[1]
    self$block2 <- make_stage(2L); cur <- cur + depths[2]
    self$block3 <- make_stage(3L); cur <- cur + depths[3]
    self$block4 <- make_stage(4L)

    # DyT stage norms
    self$norm1 <- DyT(embed_dims[1])
    self$norm2 <- DyT(embed_dims[2])
    self$norm3 <- DyT(embed_dims[3])
    self$norm4 <- DyT(embed_dims[4])

    # Decoder linear projections
    self$linear_c4 <- LinearMLP(embed_dims[4], decoder_dim)
    self$linear_c3 <- LinearMLP(embed_dims[3], decoder_dim)
    self$linear_c2 <- LinearMLP(embed_dims[2], decoder_dim)
    self$linear_c1 <- LinearMLP(embed_dims[1], decoder_dim)

    # Efficient skip connections (3: c4->c3, c3->c2, c2->c1)
    self$skip_fusions <- nn_module_list(
      list(EfficientSkipConnection(decoder_dim),
           EfficientSkipConnection(decoder_dim),
           EfficientSkipConnection(decoder_dim))
    )

    # Shared final-decoder components
    self$linear_fuse <- nn_conv3d(decoder_dim, decoder_dim, kernel_size = 1L)
    self$dropout     <- nn_dropout3d(drop_rate)

    # Head-specific prediction layers:
    #   "segmentation"   — 2-stage Conv3d  (woodcls / stemcls models)
    #   "detection_stem" — 3-stage Conv2d after collapsing Z → 2D map
    #   "regression"     — 3-stage Conv3d outputting out_chans (dx,dy)
    if (head_type == "segmentation") {
      # Two-stage: Conv3d(dec,dec//2,3,pad=1) → GELU → Conv3d(dec//2,nc,1)
      # State-dict keys: linear_pred.0 / .2
      self$linear_pred <- nn_sequential(
        nn_conv3d(decoder_dim, decoder_dim %/% 2L, kernel_size = 3L, padding = 1L),
        nn_gelu(),
        nn_conv3d(decoder_dim %/% 2L, num_classes, kernel_size = 1L)
      )
    } else if (head_type == "detection_stem") {
      # Three-stage Conv2d applied after AdaptiveMaxPool3d collapses Z to 1.
      # State-dict keys: linear_pred.0 / .2 / .4
      self$linear_pred <- nn_sequential(
        nn_conv2d(decoder_dim,          decoder_dim %/% 2L,
                  kernel_size = 3L, padding = 1L),
        nn_gelu(),
        nn_conv2d(decoder_dim %/% 2L,   decoder_dim %/% 4L,
                  kernel_size = 3L, padding = 1L),
        nn_gelu(),
        nn_conv2d(decoder_dim %/% 4L,   num_classes, kernel_size = 1L)
      )
    } else if (head_type == "regression") {
      # Three-stage Conv3d producing per-voxel offset prediction.
      # State-dict keys: linear_pred.0 / .2 / .4
      self$linear_pred <- nn_sequential(
        nn_conv3d(decoder_dim,          decoder_dim %/% 2L,
                  kernel_size = 3L, padding = 1L),
        nn_gelu(),
        nn_conv3d(decoder_dim %/% 2L,   decoder_dim %/% 4L,
                  kernel_size = 3L, padding = 1L),
        nn_gelu(),
        nn_conv3d(decoder_dim %/% 4L,   out_chans, kernel_size = 1L)
      )
    } else {
      stop(sprintf("Unknown head_type '%s'. Use 'segmentation', 'detection_stem', or 'regression'.",
                   head_type))
    }
  },

  forward_features = function(x) {
    B <- x$size(1)
    outs <- list()

    # Stage 1
    pe <- self$patch_embed1(x); x <- pe$x; D <- pe$D; H <- pe$H; W <- pe$W
    for (i in seq_along(self$block1)) x <- self$block1[[i]](x, D, H, W)
    x <- self$norm1(x)
    x <- x$reshape(c(B, D, H, W, -1L))$permute(c(1, 5, 2, 3, 4))$contiguous()
    outs[[1]] <- x

    # Stage 2
    pe <- self$patch_embed2(x); x <- pe$x; D <- pe$D; H <- pe$H; W <- pe$W
    for (i in seq_along(self$block2)) x <- self$block2[[i]](x, D, H, W)
    x <- self$norm2(x)
    x <- x$reshape(c(B, D, H, W, -1L))$permute(c(1, 5, 2, 3, 4))$contiguous()
    outs[[2]] <- x

    # Stage 3
    pe <- self$patch_embed3(x); x <- pe$x; D <- pe$D; H <- pe$H; W <- pe$W
    for (i in seq_along(self$block3)) x <- self$block3[[i]](x, D, H, W)
    x <- self$norm3(x)
    x <- x$reshape(c(B, D, H, W, -1L))$permute(c(1, 5, 2, 3, 4))$contiguous()
    outs[[3]] <- x

    # Stage 4
    pe <- self$patch_embed4(x); x <- pe$x; D <- pe$D; H <- pe$H; W <- pe$W
    for (i in seq_along(self$block4)) x <- self$block4[[i]](x, D, H, W)
    x <- self$norm4(x)
    x <- x$reshape(c(B, D, H, W, -1L))$permute(c(1, 5, 2, 3, 4))$contiguous()
    outs[[4]] <- x

    outs
  },

  forward = function(x) {
    sz <- x$size()
    d_out <- sz[3]; h_out <- sz[4]; w_out <- sz[5]

    feats <- self$forward_features(x)
    c1 <- feats[[1]]; c2 <- feats[[2]]; c3 <- feats[[3]]; c4 <- feats[[4]]
    n  <- c4$size(1)

    # Decoder — inline proj_feat at each site (nested-function dispatch in
    # nn_module forward causes "could not find function 'fn'" in R torch).
    sz4 <- c4$size()
    x4 <- self$linear_c4(c4)$permute(c(1, 3, 2))$reshape(c(n, -1L, sz4[3], sz4[4], sz4[5]))

    sz3 <- c3$size()
    p3 <- self$linear_c3(c3)$permute(c(1, 3, 2))$reshape(c(n, -1L, sz3[3], sz3[4], sz3[5]))
    x <- nnf_interpolate(x4, size = c3$size()[3:5],
                         mode = "trilinear", align_corners = FALSE)
    x <- self$skip_fusions[[1]](x, p3)

    sz2 <- c2$size()
    p2 <- self$linear_c2(c2)$permute(c(1, 3, 2))$reshape(c(n, -1L, sz2[3], sz2[4], sz2[5]))
    x <- nnf_interpolate(x, size = c2$size()[3:5],
                         mode = "trilinear", align_corners = FALSE)
    x <- self$skip_fusions[[2]](x, p2)

    sz1 <- c1$size()
    p1 <- self$linear_c1(c1)$permute(c(1, 3, 2))$reshape(c(n, -1L, sz1[3], sz1[4], sz1[5]))
    x <- nnf_interpolate(x, size = c1$size()[3:5],
                         mode = "trilinear", align_corners = FALSE)
    x <- self$skip_fusions[[3]](x, p1)

    x <- self$linear_fuse(x)
    x <- self$dropout(x)

    # Upsample to original voxel resolution BEFORE prediction head
    # (matches upstream order: interpolate → linear_pred, align_corners=TRUE)
    x <- nnf_interpolate(x, size = c(d_out, h_out, w_out),
                         mode = "trilinear", align_corners = TRUE)

    if (self$head_type == "detection_stem") {
      # Collapse spatial Z-dim (dim 3) to 1 via max, then apply 2D Conv head.
      # Upstream: x = adaptive_pool(x); x = x.squeeze(2); x = linear_pred(x)
      x <- torch_amax(x, dim = 3L, keepdim = TRUE)$squeeze(3L)  # (B, C, H, W)
    }

    self$linear_pred(x)$to(dtype = torch_float())
  }
)

# Expose as a constructor function (called by load_treeaibox_model when
# config$model$type matches "ESegFormer" / "ESeg" / "esegformer").
# head_type controls the prediction head; see ESegformer3D comments above.
# in_chans: 1 for all models except treeoff (in_chans=2).
build_esegformer3d <- function(config, num_classes_override = NULL,
                               head_type = "segmentation", out_chans = 2L,
                               in_chans = 1L) {
  cfg <- config$model
  ESegformer3D(
    block3d_size   = cfg$voxel_number_in_block,
    in_chans       = as.integer(in_chans),
    num_classes    = num_classes_override %||% (cfg$num_classes + 1L),
    patch_size     = cfg$patch_size,
    decoder_dim    = cfg$decoder_dim,
    embed_dims     = cfg$channel_dims,
    num_heads      = cfg$num_heads,
    mlp_ratios     = cfg$MLP_ratios,
    qkv_bias       = isTRUE(cfg$qkv_bias),
    depths         = cfg$depths,
    sr_ratios      = cfg$SR_ratios,
    drop_rate      = 0,
    drop_path_rate = 0,
    head_type      = head_type,
    out_chans      = as.integer(out_chans)
  )
}
