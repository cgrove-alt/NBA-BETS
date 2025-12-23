import { ChevronDown } from 'lucide-react';
import { formatMatchup, formatGameTime } from '../../lib/utils';
import type { Game } from '../../lib/types';

interface GameSelectorProps {
  games: Game[];
  selectedGameId: string | null;
  onSelectGame: (gameId: string) => void;
  loading?: boolean;
}

export function GameSelector({ games, selectedGameId, onSelectGame, loading }: GameSelectorProps) {
  return (
    <div className="relative">
      <select
        value={selectedGameId || ''}
        onChange={(e) => onSelectGame(e.target.value)}
        disabled={loading || games.length === 0}
        className="
          appearance-none w-full
          bg-bg-secondary border border-border rounded-lg
          px-4 py-3 pr-10
          text-text-primary text-sm font-medium
          cursor-pointer
          hover:border-accent-primary focus:border-accent-primary focus:outline-none
          disabled:opacity-50 disabled:cursor-not-allowed
          transition-colors
        "
      >
        {games.length === 0 ? (
          <option value="">No games today</option>
        ) : (
          <>
            <option value="">Select a game...</option>
            {games.map((game) => (
              <option key={game.game_id} value={game.game_id}>
                {formatMatchup(game.home_team.abbreviation, game.visitor_team.abbreviation)}
                {game.game_time ? ` - ${formatGameTime(game.game_time)}` : ''}
              </option>
            ))}
          </>
        )}
      </select>
      <ChevronDown
        size={18}
        className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
      />
    </div>
  );
}
