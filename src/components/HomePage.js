import { useNavigate } from 'react-router-dom';

function HomePage() {
    const navigate = useNavigate();

    const handleNavigation = () => {
        navigate('/some-path');
    };

    return (
        <div>
            <h1>Home Page</h1>
            <button onClick={handleNavigation}>Go to Some Path</button>
        </div>
    );
}

export default HomePage; 