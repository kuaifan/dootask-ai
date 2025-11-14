import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { MCPConfig } from "@/data/mcp-config"
import type { AIBotItem } from "@/data/aibots"
import { useI18n } from "@/lib/i18n-context"

export interface MCPListCardProps {
  mcps: MCPConfig[]
  bots: AIBotItem[]
  allModels: Record<string, string> // key: modelValue, value: modelLabel
  onEdit: (mcp: MCPConfig) => void
  onDelete: (name: string) => void
  onAdd: () => void
}

export const MCPListCard = ({
  mcps,
  allModels,
  onEdit,
  onDelete,
  onAdd,
}: MCPListCardProps) => {
  const { t } = useI18n()

  const getModelLabel = (modelValue: string) => {
    return allModels[modelValue] ?? modelValue
  }

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="text-lg font-semibold">
          {t("mcp.title")}
        </CardTitle>
        <Button onClick={onAdd} size="sm">
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
          <div className="space-y-3">
            {mcps.map((mcp) => (
              <div
                key={mcp.name}
                className="flex items-start justify-between rounded-lg border p-4 hover:bg-muted/50 transition-colors"
              >
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2 min-h-8">
                    <h4 className="font-medium text-sm">{mcp.name}</h4>
                    <Badge variant={mcp.enabled ? "default" : "secondary"} className="text-xs">
                      {mcp.enabled ? t("mcp.statusEnabled") : t("mcp.statusDisabled")}
                    </Badge>
                  </div>
                  {mcp.supportedModels.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {mcp.supportedModels.slice(0, 2).map((modelValue) => (
                        <Badge
                          key={modelValue}
                          variant="secondary"
                          className="text-xs"
                        >
                          {getModelLabel(modelValue)}
                        </Badge>
                      ))}
                      {mcp.supportedModels.length > 2 && (
                        <Badge variant="secondary" className="text-xs">
                          +{mcp.supportedModels.length - 2}
                        </Badge>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onEdit(mcp)}
                  >
                    {t("mcp.edit")}
                  </Button>
                  {!mcp.isSystem && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onDelete(mcp.name)}
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
