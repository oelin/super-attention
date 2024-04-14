"""Super attention."""

class SuperAttention(nn.Module):
    """Super attention.

    Example
    -------
    >>> module = SuperAttention(
    ...     embedding_dimension=256,
    ...     sequence_length=1024,
    ... )
    >>> x = torch.randn((1, 1024, 256))
    >>> x = module(x)  # Shape: (1, 1024, 256).
    """

    def __init__(
        self, 
        embedding_dimension: int,
        sequence_length: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        sequence_length : int
            The sequence length.
        """

        super().__init__()

        self.project_query = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.project_value = nn.Linear(
            in_features=sequence_length,
            out_features=sequence_length,
            bias=False,
        )

        self.project_output = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        query = x @ self.project_query.weight
        value = self.project_value.weight @ x
        score = torch.einsum('bxe,bye->bxy', query, x)
        score = torch.softmax(score / math.sqrt(x.size(-1)), dim=-1)
        output = self.project_output(score @ value)

        return output
