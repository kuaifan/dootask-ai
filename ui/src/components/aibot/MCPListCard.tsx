import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { MCPConfig } from "@/data/mcp-config"
import type { AIBotItem } from "@/data/aibots"
import { useI18n } from "@/lib/i18n-context"
import { Plus, Plug, Settings } from "lucide-react"

export interface MCPListCardProps {
  mcps: MCPConfig[]
  bots: AIBotItem[]
  onEdit: (mcp: MCPConfig) => void
  onDelete: (mcp: MCPConfig) => void
  onAdd: () => void
}

export const MCPListCard = ({
  mcps,
  onEdit,
  onDelete,
  onAdd,
}: MCPListCardProps) => {
  const { t } = useI18n()

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <div className="flex items-center gap-2">
          <Plug className="h-5 w-5" />
          <CardTitle className="text-base">{t("mcp.title")}</CardTitle>
        </div>
        <Button size="sm" onClick={onAdd}>
          <Plus />
          {t("mcp.addButton")}
        </Button>
      </CardHeader>
      <CardContent>
        {mcps.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <p className="text-sm text-muted-foreground mb-4">
              {t("mcp.empty")}
            </p>
            <Button onClick={onAdd} variant="outline" size="sm">
              {t("mcp.addButton")}
            </Button>
          </div>
        ) : (
          <div className="divide-y mt-1">
            {mcps.map((mcp) => (
              <div
                key={mcp.id}
                className="flex items-start justify-between py-3 first:pt-0 last:pb-0"
              >
                <div className="flex-1 space-y-3">
                  <h4 className="font-medium text-sm">{mcp.name}</h4>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">{t("mcp.enabled")}:</span>
                    <Badge variant={mcp.enabled ? "default" : "secondary"}>
                      {mcp.enabled ? t("mcp.statusEnabled") : t("mcp.statusDisabled")}
                    </Badge>
                  </div>
                  {mcp.supportedModels.length > 0 && (
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm text-muted-foreground">{t("mcp.supportedModels")}:</span>
                      {mcp.supportedModels.slice(0, 3).map((model) => (
                        <Badge
                          key={model.id}
                          variant="outline"
                          className="text-xs"
                        >
                          {model.name}
                        </Badge>
                      ))}
                      {mcp.supportedModels.length > 3 && (
                        <Badge variant="outline" className="text-xs">
                          +{mcp.supportedModels.length - 3}
                        </Badge>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1 ml-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onEdit(mcp)}
                  >
                    <Settings />
                    {t("mcp.edit")}
                  </Button>
                  {!mcp.isSystem && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onDelete(mcp)}
                    >
                      {t("mcp.delete")}
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
