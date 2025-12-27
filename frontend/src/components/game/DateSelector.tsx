import { useMemo } from 'react';

interface DateOption {
  date: string;      // YYYY-MM-DD format for API
  label: string;     // Display label (e.g., "Thu 26")
  isToday: boolean;
}

interface DateSelectorProps {
  selectedDate: string;
  onSelectDate: (date: string) => void;
}

/**
 * Generate date options for today through 3 days in the future.
 * Uses Eastern timezone to match NBA schedule.
 */
function generateDateOptions(): DateOption[] {
  const options: DateOption[] = [];
  const now = new Date();

  // Create dates for today + next 3 days
  for (let i = 0; i < 4; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() + i);

    // Format as YYYY-MM-DD for API
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const dateStr = `${year}-${month}-${day}`;

    // Format display label (e.g., "Thu 26")
    const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
    const dayNum = date.getDate();
    const label = `${dayName} ${dayNum}`;

    options.push({
      date: dateStr,
      label,
      isToday: i === 0,
    });
  }

  return options;
}

export function DateSelector({ selectedDate, onSelectDate }: DateSelectorProps) {
  const dateOptions = useMemo(() => generateDateOptions(), []);

  return (
    <div className="flex gap-2">
      {dateOptions.map((option) => (
        <button
          key={option.date}
          onClick={() => onSelectDate(option.date)}
          className={`
            px-4 py-2 rounded-lg text-sm font-medium
            transition-colors
            ${selectedDate === option.date
              ? 'bg-accent-primary text-white'
              : 'bg-bg-secondary border border-border text-text-primary hover:border-accent-primary'
            }
          `}
        >
          <span>{option.label}</span>
          {option.isToday && (
            <span className="ml-1 text-xs opacity-75">Today</span>
          )}
        </button>
      ))}
    </div>
  );
}

/**
 * Get today's date in YYYY-MM-DD format.
 */
export function getTodayDate(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}
