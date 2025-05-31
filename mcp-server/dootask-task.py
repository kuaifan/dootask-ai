from fastmcp import FastMCP
import json
import httpx
from fastmcp.server.dependencies import get_http_headers

mcp = FastMCP("dootask-task")

def get_token() -> str:
    headers = get_http_headers()
    # Get authorization header
    auth_header = headers.get("token", "")
    if not auth_header:
        return "请登录"
    return auth_header
@mcp.tool()
async def get_all_tasks(
     archived: str = "no", deleted: str = "no", sort_type: str = None, sort: str="desc"
) -> dict:
    """
    获取用户任务列表
    Args:
        archived: 归档状态(no 未归档,yes 已归档,all 所有)
        deleted: 删除状态(no 未删除,yes 已删除,all 所有)
        sort_type: 排序方式
            - sorts[complete_at] 完成时间asc|desc
            - sorts[archived_at] 归档时间asc|desc
            - sorts[end_at] 到期时间asc|desc
    Returns:
        将数据以表格形式返回,注意显示任务ID
    """

    async with httpx.AsyncClient() as client:
        params = {"archived": archived, "deleted": deleted}
        if sort_type:
            params[sort_type] = sort
        response = await client.get(
            "https://t.hitosea.com/api/project/task/lists",
            headers={"token": get_token()},
            params=params,
        )
        result = {"data":[]}
        if response and response.json().get("ret") == 1:
            for item in response.json()["data"]["data"]:
                task = {}
                task["id"] = item.get("id")
                task["name"] = item.get("name")
                task["start_at"] = item.get("start_at")
                task["end_at"] = item.get("end_at")
                task["p_name"] = item.get("p_name")
                result["data"].append(task)
        # print(response.json())
        return result


@mcp.tool()
async def get_task_detail(taskId: int, archived: str = "no") -> dict:
    """
    根据任务id获取任务详细信息
    Args:
        taskId: 任务id
        archived: (no 未归档,yes 已归档,all 所有)
    Return:
        返回json格式数据
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://t.hitosea.com/api/project/task/one",
            headers={"token": get_token()},
            params={"task_id": taskId, "archived": archived},
        )
        print(response.json())
        if response and response.json().get("ret") == 1:
            return response.json().get("data")
        else:
            return "查询失败"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=9000)