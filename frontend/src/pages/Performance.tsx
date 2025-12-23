import { BarChart2 } from 'lucide-react';
import { Card, CardContent } from '../components/ui/Card';

export function Performance() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Performance</h1>
        <p className="text-sm text-text-secondary mt-1">Model accuracy and historical results</p>
      </div>

      <Card>
        <CardContent className="py-12">
          <div className="text-center">
            <BarChart2 className="mx-auto text-text-muted mb-4" size={48} />
            <h2 className="text-lg font-medium text-text-primary mb-2">Coming Soon</h2>
            <p className="text-text-secondary">
              Performance analytics are under development.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
