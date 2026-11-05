from mcp.server.mcpserver import MCPServer as FastMCP
mcp = FastMCP("probe")
@mcp.tool()
def echo(text: str) -> str:
    """Echo text"""
    return text
if __name__ == "__main__":
    mcp.run()  # stdio
