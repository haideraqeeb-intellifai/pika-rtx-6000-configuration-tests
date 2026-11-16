def forward(self, x, rope_freqs=None, transformer_options={}):
    s = x.shape[0]
    batch = transformer_options.get("_h3_batch_size", 1)
    assert s % batch == 0
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    q = q.reshape(batch, s // batch, self.heads, self.head_dim).transpose(1, 2)
    k = k.reshape(batch, s // batch, self.heads, self.head_dim).transpose(1, 2)
    v = v.reshape(batch, s // batch, self.heads, self.head_dim).transpose(1, 2)
    out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
    return self.out_proj(out.reshape(s, self.heads * self.head_dim))
