class ConvTranpose2D(nn.Conv):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            use_bias=True,
            *,
            key,
            **kwargs,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )

    def __call__(
            self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**
        - `x`: The input. Should be a JAX array of shape `(in_channels, dim_1, ..., dim_N)`, where
            `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array of shape `(out_channels, new_dim_1, ..., new_dim_N)`.
        """
        input_dim = x.shape[1] + (x.shape[1] - 1) * (self.stride[0] - 1)
        pad_row = jnp.zeros((x.shape[0], input_dim, x.shape[1]))
        pad_row = pad_row.at[:, ::2].set(x)
        pad_row_col = jnp.zeros((x.shape[0], input_dim, input_dim))
        pad_row_col = pad_row_col.at[:, :, ::2].set(pad_row)
        padding_amount = self.kernel_size[0] - self.padding[0][0] - 1
        pad_row_col_out = jnp.pad(pad_row_col, [(0, 0), (padding_amount, padding_amount),
                                                (padding_amount, padding_amount)])

        x = pad_row_col_out
        breakpoint()
        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )
        x = jnp.expand_dims(x, axis=0)
        x = conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=(1, 1),
            padding=(0, 0),
            rhs_dilation=self.dilation
        )
        if self.use_bias:
            x += jnp.broadcast_to(self.bias, x.shape)
        x = jnp.squeeze(x, axis=0)
        return x